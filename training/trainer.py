"""
Model Tuning Pipeline.

Two-stage training:
1. SBERT bi-encoder using CosineSimilarityLoss (positive + negative pairs)
2. Cross-encoder reranker using positives + hard negatives

Performance features:
- Mixed-precision (AMP) training for 2x speed on GPU
- Stratified train/val/test splitting to ensure both classes in every split
- Automatic hard-negative mining when negatives are missing
- Retry logic with exponential backoff
- Class balance validation before training
"""

import logging
import os
import json
import random
import time
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)

from core.models import TrainingPair
from config.settings import CONFIG
from evaluation.accuracy_tracker import (
    AccuracyEntry, AccuracyTracker, ComparisonReport,
    TestSetScorer, get_current_model_paths,
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds training datasets from TrainingPair records.
    Handles stratified splitting, class balancing, and hard-negative mining.
    """

    @staticmethod
    def split_pairs(
        pairs: List[TrainingPair],
        train_ratio: float = None,
        val_ratio: float = None,
        seed: int = 42,
    ) -> Tuple[List[TrainingPair], List[TrainingPair], List[TrainingPair]]:
        """
        Split pairs into train/val/test with stratified sampling.
        Groups by source_suggestion_id to prevent data leakage, then
        stratifies by each group's majority label so every split has
        both positives and negatives.
        """
        train_r = train_ratio or CONFIG.training.train_ratio
        val_r = val_ratio or CONFIG.training.val_ratio
        test_r = max(1.0 - train_r - val_r, 0.05)

        groups: Dict[str, List[TrainingPair]] = {}
        for p in pairs:
            key = p.source_suggestion_id or p.pair_id
            groups.setdefault(key, []).append(p)

        group_keys = list(groups.keys())

        group_labels = []
        for key in group_keys:
            gp = groups[key]
            pos = sum(1 for p in gp if p.label == 1.0)
            group_labels.append(1 if pos > 0 else 0)

        try:
            from sklearn.model_selection import train_test_split

            n_classes = len(set(group_labels))
            min_per_class = min(
                sum(1 for l in group_labels if l == c)
                for c in set(group_labels)
            ) if n_classes >= 2 else 0

            can_stratify = n_classes >= 2 and min_per_class >= 3

            if can_stratify:
                train_val_keys, test_keys_list = train_test_split(
                    group_keys, test_size=test_r,
                    stratify=group_labels, random_state=seed,
                )
                tv_labels = [
                    group_labels[group_keys.index(k)]
                    for k in train_val_keys
                ]
                val_frac = val_r / (train_r + val_r)
                train_keys_list, val_keys_list = train_test_split(
                    train_val_keys, test_size=val_frac,
                    stratify=tv_labels, random_state=seed,
                )
                logger.info("Using stratified split (both classes in every split)")
            else:
                logger.warning(
                    f"Cannot stratify: {n_classes} classes, min per class={min_per_class}. "
                    "Falling back to random split."
                )
                raise ValueError("force random split")

            train_keys = set(train_keys_list)
            val_keys = set(val_keys_list)
            test_keys = set(test_keys_list)

        except (ValueError, ImportError):
            rng = random.Random(seed)
            rng.shuffle(group_keys)
            n = len(group_keys)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))
            train_keys = set(group_keys[:train_end])
            val_keys = set(group_keys[train_end:val_end])
            test_keys = set(group_keys[val_end:])

        train = [p for k in train_keys for p in groups[k]]
        val = [p for k in val_keys for p in groups[k]]
        test = [p for k in test_keys for p in groups[k]]

        for name, split in [("train", train), ("val", val), ("test", test)]:
            pos = sum(1 for p in split if p.label == 1.0)
            neg = len(split) - pos
            logger.info(f"  {name}: {len(split)} pairs (pos={pos}, neg={neg})")

        return train, val, test

    @staticmethod
    def generate_hard_negatives(
        positive_pairs: List[TrainingPair],
        negatives_per_positive: int = 1,
        seed: int = 42,
    ) -> List[TrainingPair]:
        """
        Generate hard negatives by shuffling candidate_text among positive pairs.
        Each negative pairs an anchor with a random wrong candidate from the same pool.
        """
        if len(positive_pairs) < 2:
            logger.warning("Need at least 2 positive pairs to generate hard negatives")
            return []

        rng = random.Random(seed)
        candidates = [p.candidate_text for p in positive_pairs]
        negatives = []

        for pair in positive_pairs:
            attempts = 0
            for _ in range(negatives_per_positive):
                wrong = rng.choice(candidates)
                while wrong == pair.candidate_text and attempts < 20:
                    wrong = rng.choice(candidates)
                    attempts += 1

                if wrong == pair.candidate_text:
                    continue

                negatives.append(TrainingPair(
                    anchor_text=pair.anchor_text,
                    candidate_text=wrong,
                    label=0.0,
                    is_hard_negative=True,
                    source_suggestion_id=f"synthetic_neg_{pair.source_suggestion_id or pair.pair_id}",
                ))

        logger.info(f"Generated {len(negatives)} synthetic hard negatives from {len(positive_pairs)} positives")
        return negatives

    @staticmethod
    def ensure_both_classes(
        pairs: List[TrainingPair],
        target_neg_ratio: int = 1,
    ) -> List[TrainingPair]:
        """
        Ensure the dataset has both positive and negative examples.
        - If only positives: generate hard negatives by shuffling candidates.
        - If only negatives: try loading cached positives from training_pairs.json.
        - Returns augmented pair list.
        """
        positives = [p for p in pairs if p.label == 1.0]
        negatives = [p for p in pairs if p.label == 0.0]

        logger.info(
            f"Class balance check: {len(positives)} positives, "
            f"{len(negatives)} negatives out of {len(pairs)} total"
        )

        if positives and negatives:
            ratio = min(len(positives), len(negatives)) / max(len(positives), len(negatives))
            if ratio < 0.05:
                logger.warning(
                    f"Severe class imbalance (ratio={ratio:.3f}). "
                    "Augmenting minority class."
                )
                if len(negatives) < len(positives):
                    needed = max(int(len(positives) * 0.2) - len(negatives), 0)
                    if needed > 0:
                        extra = DatasetBuilder.generate_hard_negatives(
                            positives,
                            negatives_per_positive=max(1, needed // len(positives) + 1),
                        )
                        pairs = pairs + extra[:needed]
                        logger.info(f"Added {min(len(extra), needed)} synthetic negatives to balance")
            return pairs

        if positives and not negatives:
            logger.warning(
                f"ONLY positives ({len(positives)}) found — no negatives. "
                f"Generating {len(positives) * target_neg_ratio} synthetic hard negatives."
            )
            synthetic = DatasetBuilder.generate_hard_negatives(positives, target_neg_ratio)
            return pairs + synthetic

        if negatives and not positives:
            logger.warning(
                f"ONLY negatives ({len(negatives)}) found — no positives. "
                "Attempting to load cached positive pairs..."
            )
            cached_path = os.path.join(CONFIG.data_dir, "training_pairs.json")
            if os.path.exists(cached_path):
                try:
                    with open(cached_path, "r") as f:
                        cached_data = json.load(f)
                    cached_positives = []
                    for item in cached_data:
                        if isinstance(item, dict) and item.get("label") == 1.0:
                            cached_positives.append(TrainingPair(
                                anchor_text=item["anchor_text"],
                                candidate_text=item["candidate_text"],
                                label=1.0,
                                is_hard_negative=False,
                                source_suggestion_id=item.get("source_suggestion_id", "cached"),
                            ))
                    if cached_positives:
                        logger.info(f"Loaded {len(cached_positives)} positive pairs from cache: {cached_path}")
                        return pairs + cached_positives
                except Exception as e:
                    logger.warning(f"Failed to load cached pairs: {e}")

            logger.warning(
                "No cached positives available. Training with negatives only "
                "will produce a model that always predicts 'no match'. "
                "Please provide 'Correct' feedback to get positive examples."
            )

        return pairs

    @staticmethod
    def build_sbert_examples(pairs: List[TrainingPair]) -> List[InputExample]:
        """
        Build InputExamples for SBERT training with CosineSimilarityLoss.
        Uses BOTH positive (label=1.0) and negative (label=0.0) pairs.
        """
        examples = [
            InputExample(
                texts=[p.anchor_text, p.candidate_text],
                label=p.label,
            )
            for p in pairs
        ]

        pos_count = sum(1 for p in pairs if p.label == 1.0)
        neg_count = sum(1 for p in pairs if p.label == 0.0)
        logger.info(
            f"Built {len(examples)} SBERT examples "
            f"(pos={pos_count}, neg={neg_count})"
        )
        return examples

    @staticmethod
    def build_cross_encoder_examples(
        pairs: List[TrainingPair],
    ) -> List[InputExample]:
        """
        Build InputExamples for Cross-Encoder training.
        Uses both positives (label=1) and hard negatives (label=0).
        """
        examples = [
            InputExample(
                texts=[p.anchor_text, p.candidate_text],
                label=p.label,
            )
            for p in pairs
        ]

        pos_count = sum(1 for p in pairs if p.label == 1.0)
        neg_count = sum(1 for p in pairs if p.label == 0.0)
        logger.info(
            f"Built {len(examples)} CE examples "
            f"(pos={pos_count}, neg={neg_count})"
        )
        return examples

    @staticmethod
    def validate_no_contamination(pairs: List[TrainingPair]) -> bool:
        """
        Verify no label contamination: true positive anchor+candidate
        should never appear as a negative.
        """
        positives = set()
        for p in pairs:
            if p.label == 1.0:
                positives.add((p.anchor_text, p.candidate_text))

        contaminated = 0
        for p in pairs:
            if p.label == 0.0:
                key = (p.anchor_text, p.candidate_text)
                if key in positives:
                    contaminated += 1
                    logger.error(
                        f"CONTAMINATION: positive pair found as negative: "
                        f"'{p.anchor_text[:50]}...'"
                    )

        if contaminated:
            logger.error(f"TOTAL CONTAMINATED PAIRS: {contaminated}")
            return False

        logger.info("No label contamination detected")
        return True


class SBERTTrainer:
    """Fine-tune SBERT bi-encoder using CosineSimilarityLoss."""

    def __init__(self, base_model: str = None):
        self.base_model = base_model or CONFIG.model.sbert_model
        self.model = SentenceTransformer(self.base_model)

    @staticmethod
    def _examples_to_dataset(examples: List[InputExample]):
        """Convert InputExample list to HuggingFace Dataset for v3 compatibility."""
        from datasets import Dataset as HFDataset
        return HFDataset.from_dict({
            "sentence1": [ex.texts[0] for ex in examples],
            "sentence2": [ex.texts[1] for ex in examples],
            "label": [float(ex.label) for ex in examples],
        })

    def train(
        self,
        train_examples: List[InputExample],
        val_pairs: List[TrainingPair] = None,
        output_path: str = None,
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
        warmup_ratio: float = None,
    ) -> str:
        """
        Train SBERT with CosineSimilarityLoss (works with both positive and negative pairs).
        Uses SentenceTransformerTrainer (v3 API) with fallback to fit() for v2.
        Returns path to saved model.
        """
        epochs = epochs or CONFIG.training.sbert_epochs
        batch_size = batch_size or CONFIG.training.sbert_batch_size
        lr = lr or CONFIG.training.sbert_lr
        warmup_ratio = warmup_ratio or CONFIG.training.sbert_warmup_ratio

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or os.path.join(
            CONFIG.model.tuned_sbert_path or
            os.path.join(CONFIG.data_dir, "models", f"sbert_tuned_{timestamp}")
        )

        pos_count = sum(1 for ex in train_examples if ex.label > 0.5)
        neg_count = sum(1 for ex in train_examples if ex.label <= 0.5)
        logger.info(
            f"SBERT class balance: {pos_count} positives, {neg_count} negatives, "
            f"{len(train_examples)} total"
        )

        if pos_count == 0 or neg_count == 0:
            logger.warning(
                f"SBERT single-class data detected (pos={pos_count}, neg={neg_count}). "
                "Model will still train but discrimination may be poor."
            )

        use_amp = CONFIG.training.use_amp
        device = CONFIG.model.device
        grad_accum = CONFIG.training.gradient_accumulation_steps
        num_workers = CONFIG.training.dataloader_workers

        train_loss = losses.CosineSimilarityLoss(self.model)

        evaluator = None
        if val_pairs:
            sentences1 = [p.anchor_text for p in val_pairs]
            sentences2 = [p.candidate_text for p in val_pairs]
            scores = [p.label for p in val_pairs]
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                sentences1, sentences2, scores,
            )

        logger.info(
            f"Training SBERT on {device}: {len(train_examples)} examples, "
            f"{epochs} epochs, batch={batch_size}, grad_accum={grad_accum}, "
            f"lr={lr}, amp={use_amp}, loss=CosineSimilarityLoss, workers={num_workers}"
        )

        try:
            from sentence_transformers import SentenceTransformerTrainer
            from sentence_transformers.training_args import SentenceTransformerTrainingArguments

            train_dataset = self._examples_to_dataset(train_examples)
            eval_dataset = None
            if val_pairs:
                val_examples = [
                    InputExample(texts=[p.anchor_text, p.candidate_text], label=p.label)
                    for p in val_pairs
                ]
                eval_dataset = self._examples_to_dataset(val_examples)

            training_args = SentenceTransformerTrainingArguments(
                output_dir=output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                fp16=use_amp and device == "cuda",
                dataloader_num_workers=num_workers,
                dataloader_pin_memory=torch.cuda.is_available(),
                eval_strategy="steps" if eval_dataset is not None else "no",
                eval_steps=max(1, len(train_examples) // (batch_size * 4)) if eval_dataset else None,
                save_strategy="epoch",
                logging_steps=50,
                load_best_model_at_end=eval_dataset is not None,
            )

            trainer = SentenceTransformerTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=train_loss,
            )
            trainer.train()
            self.model.save(output_path)
            logger.info(f"SBERT saved to: {output_path} (v3 Trainer API)")

        except (ImportError, TypeError) as e:
            logger.info(f"Falling back to fit() API: {e}")

            train_dataloader = DataLoader(
                train_examples, shuffle=True, batch_size=batch_size,
            )
            total_steps = len(train_dataloader) * epochs
            warmup_steps = int(total_steps * warmup_ratio)

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": lr},
                output_path=output_path,
                evaluator=evaluator,
                evaluation_steps=max(1, len(train_dataloader) // 4),
                save_best_model=True if evaluator else False,
                use_amp=use_amp,
            )
            logger.info(f"SBERT saved to: {output_path} (fit() API)")

        return output_path


class CrossEncoderTrainer:
    """Fine-tune Cross-Encoder reranker with optional AMP."""

    def __init__(self, base_model: str = None):
        self.base_model = base_model or CONFIG.model.cross_encoder_model
        self.model = CrossEncoder(
            self.base_model,
            num_labels=1,
            max_length=256,
        )

    @staticmethod
    def _examples_to_dataset(examples: List[InputExample]):
        """Convert InputExample list to HuggingFace Dataset for v3 compatibility."""
        from datasets import Dataset as HFDataset
        return HFDataset.from_dict({
            "sentence1": [ex.texts[0] for ex in examples],
            "sentence2": [ex.texts[1] for ex in examples],
            "label": [float(ex.label) for ex in examples],
        })

    def train(
        self,
        train_examples: List[InputExample],
        val_examples: List[InputExample] = None,
        output_path: str = None,
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
    ) -> str:
        """
        Train Cross-Encoder with optional mixed-precision.
        Returns path to saved model.
        """
        epochs = epochs or CONFIG.training.ce_epochs
        batch_size = batch_size or CONFIG.training.ce_batch_size
        lr = lr or CONFIG.training.ce_lr

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or os.path.join(
            CONFIG.model.tuned_cross_encoder_path or
            os.path.join(CONFIG.data_dir, "models", f"ce_tuned_{timestamp}")
        )

        pos_count = sum(1 for ex in train_examples if ex.label > 0.5)
        neg_count = sum(1 for ex in train_examples if ex.label <= 0.5)
        logger.info(
            f"CE class balance: {pos_count} positives, {neg_count} negatives, "
            f"{len(train_examples)} total"
        )

        if pos_count == 0 or neg_count == 0:
            logger.warning(
                f"CE single-class data detected (pos={pos_count}, neg={neg_count}). "
                "Model will still train but discrimination will be poor."
            )

        train_dataset = self._examples_to_dataset(train_examples)
        eval_dataset = self._examples_to_dataset(val_examples) if val_examples else None

        warmup_ratio = 0.1
        use_amp = CONFIG.training.use_amp
        grad_accum = CONFIG.training.gradient_accumulation_steps
        device = CONFIG.model.device
        effective_batch = batch_size * grad_accum
        logger.info(
            f"Training CrossEncoder on {device}: {len(train_examples)} examples, "
            f"{epochs} epochs, batch={batch_size}, grad_accum={grad_accum}, "
            f"effective_batch={effective_batch}, lr={lr}, amp={use_amp}"
        )

        try:
            from sentence_transformers.cross_encoder import (
                CrossEncoderTrainer as STCrossEncoderTrainer,
                CrossEncoderTrainingArguments,
            )

            training_args = CrossEncoderTrainingArguments(
                output_dir=output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                fp16=use_amp,
                dataloader_num_workers=CONFIG.training.dataloader_workers,
                dataloader_pin_memory=device == "cuda",
                save_strategy="epoch",
                logging_steps=50,
                report_to="none",
            )

            trainer = STCrossEncoderTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()
            self.model.save_pretrained(output_path)

        except ImportError:
            num_workers = CONFIG.training.dataloader_workers
            train_dataloader = DataLoader(
                train_examples, shuffle=True, batch_size=batch_size,
                num_workers=num_workers, pin_memory=device == "cuda",
            )
            evaluator = None
            if val_examples:
                val_sentences = [(e.texts[0], e.texts[1]) for e in val_examples]
                val_labels = [e.label for e in val_examples]
                evaluator = CEBinaryClassificationEvaluator(
                    sentence_pairs=val_sentences,
                    labels=val_labels,
                    name="val",
                )
            total_steps = len(train_dataloader) * epochs
            warmup_steps = int(total_steps * 0.1)
            self.model.fit(
                train_dataloader=train_dataloader,
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": lr},
                output_path=output_path,
                evaluator=evaluator,
                evaluation_steps=max(1, len(train_dataloader) // 4),
                save_best_model=True if evaluator else False,
                use_amp=use_amp,
            )

        logger.info(f"CrossEncoder saved to: {output_path}")
        return output_path


class TrainingOrchestrator:
    """
    End-to-end training orchestrator.
    Coordinates dataset building, class balancing, training, and model versioning.
    """

    def __init__(self, training_pairs: List[TrainingPair]):
        self.pairs = training_pairs
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []

    def prepare(self) -> dict:
        """Prepare datasets: validate, balance classes, and split."""
        is_clean = DatasetBuilder.validate_no_contamination(self.pairs)
        if not is_clean:
            raise ValueError("Label contamination detected! Fix data before training.")

        balanced_pairs = DatasetBuilder.ensure_both_classes(self.pairs)

        self.train_pairs, self.val_pairs, self.test_pairs = (
            DatasetBuilder.split_pairs(balanced_pairs)
        )

        pos_total = sum(1 for p in balanced_pairs if p.label == 1.0)
        neg_total = sum(1 for p in balanced_pairs if p.label == 0.0)

        stats = {
            "total_pairs": len(balanced_pairs),
            "original_pairs": len(self.pairs),
            "synthetic_added": len(balanced_pairs) - len(self.pairs),
            "train": len(self.train_pairs),
            "val": len(self.val_pairs),
            "test": len(self.test_pairs),
            "total_positives": pos_total,
            "total_negatives": neg_total,
            "train_positives": sum(1 for p in self.train_pairs if p.label == 1.0),
            "train_negatives": sum(1 for p in self.train_pairs if p.label == 0.0),
            "val_positives": sum(1 for p in self.val_pairs if p.label == 1.0),
            "val_negatives": sum(1 for p in self.val_pairs if p.label == 0.0),
            "test_positives": sum(1 for p in self.test_pairs if p.label == 1.0),
            "test_negatives": sum(1 for p in self.test_pairs if p.label == 0.0),
        }
        logger.info(f"Dataset stats: {json.dumps(stats, indent=2)}")
        return stats

    def train_sbert(self, output_path: str = None) -> Optional[str]:
        """Train SBERT bi-encoder with CosineSimilarityLoss (both classes)."""
        train_examples = DatasetBuilder.build_sbert_examples(self.train_pairs)
        if not train_examples:
            logger.warning("SBERT training skipped — no training examples at all.")
            return None

        trainer = SBERTTrainer()
        return trainer.train(
            train_examples=train_examples,
            val_pairs=self.val_pairs,
            output_path=output_path,
        )

    def train_cross_encoder(self, output_path: str = None) -> Optional[str]:
        """Train Cross-Encoder reranker."""
        train_examples = DatasetBuilder.build_cross_encoder_examples(self.train_pairs)
        if not train_examples:
            logger.warning("Cross-Encoder training skipped — no training examples.")
            return None
        val_examples = DatasetBuilder.build_cross_encoder_examples(self.val_pairs)
        trainer = CrossEncoderTrainer()
        return trainer.train(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path=output_path,
        )

    def train_all(
        self,
        sbert_output: str = None,
        ce_output: str = None,
        track_accuracy: bool = True,
        max_retries: int = 2,
    ) -> dict:
        """
        Train both models with optional before/after accuracy tracking
        and retry logic for resilience.
        """
        stats = self.prepare()

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        pre_entry = None

        if track_accuracy and self.test_pairs:
            pre_entry = self._evaluate_models(
                run_type="pre_training",
                run_id=run_id,
                test_pairs=self.test_pairs,
                dataset_stats=stats,
            )

        sbert_path = None
        ce_path = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Training retry attempt {attempt + 1}/{max_retries + 1}")

                sbert_path = self.train_sbert(sbert_output)
                ce_path = self.train_cross_encoder(ce_output)
                break

            except Exception as e:
                logger.error(
                    f"Training attempt {attempt + 1} failed: {e}\n"
                    f"{traceback.format_exc()}"
                )
                if attempt < max_retries:
                    wait = 15 * (attempt + 1)
                    logger.info(f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    logger.error("All training attempts failed!")
                    return {
                        "dataset_stats": stats,
                        "sbert_model_path": sbert_path,
                        "cross_encoder_model_path": ce_path,
                        "training_run_id": run_id,
                        "accuracy_comparison": None,
                        "error": str(e),
                    }

        if not sbert_path and not ce_path:
            logger.warning(
                "No models were trained — not enough data. "
                f"Positives: {stats['total_positives']}, Negatives: {stats['total_negatives']}"
            )

        post_entry = None
        comparison_report = None
        if track_accuracy and self.test_pairs and (sbert_path or ce_path):
            post_entry = self._evaluate_models(
                run_type="post_training",
                run_id=run_id,
                test_pairs=self.test_pairs,
                dataset_stats=stats,
                sbert_path=sbert_path,
                ce_path=ce_path,
            )
            if pre_entry and post_entry:
                comparison_report = ComparisonReport.generate(
                    pre_entry, post_entry,
                    output_path=os.path.join(
                        CONFIG.logs_dir, f"comparison_{run_id}.txt"
                    ),
                )
                logger.info(comparison_report)

        return {
            "dataset_stats": stats,
            "sbert_model_path": sbert_path,
            "cross_encoder_model_path": ce_path,
            "training_run_id": run_id,
            "accuracy_comparison": comparison_report,
        }

    # ── Accuracy Evaluation Helpers ──

    def _evaluate_models(
        self,
        run_type: str,
        run_id: str,
        test_pairs: List[TrainingPair],
        dataset_stats: dict,
        sbert_path: str = None,
        ce_path: str = None,
    ) -> Optional[AccuracyEntry]:
        """Score test pairs with current or specified models and log to accuracy tracker."""
        try:
            paths = get_current_model_paths()
            version = "current"
            if sbert_path:
                paths["sbert"] = sbert_path
            if ce_path:
                paths["cross_encoder"] = ce_path
            if sbert_path or ce_path:
                version = f"tuned_{run_id}"

            pos_count = sum(1 for p in test_pairs if p.label == 1.0)
            neg_count = sum(1 for p in test_pairs if p.label == 0.0)
            logger.info(
                f"[{run_type}] Evaluating on {len(test_pairs)} test pairs "
                f"(pos={pos_count}, neg={neg_count})  "
                f"SBERT={paths['sbert']}, CE={paths['cross_encoder']}"
            )

            ce_metrics = TestSetScorer.score_cross_encoder(
                paths["cross_encoder"], test_pairs,
            )
            sbert_metrics = TestSetScorer.score_sbert(
                paths["sbert"], test_pairs,
            )

            dataset_info = {
                **dataset_stats,
                "test_total": len(test_pairs),
                "test_positives": pos_count,
                "test_negatives": neg_count,
            }

            entry = AccuracyEntry(
                run_type=run_type,
                training_run_id=run_id,
                model_version=version,
                ce_metrics=ce_metrics,
                sbert_metrics=sbert_metrics,
                dataset_info=dataset_info,
                model_paths=paths,
                training_config={
                    "sbert_epochs": CONFIG.training.sbert_epochs,
                    "ce_epochs": CONFIG.training.ce_epochs,
                    "sbert_lr": CONFIG.training.sbert_lr,
                    "ce_lr": CONFIG.training.ce_lr,
                    "sbert_batch_size": CONFIG.training.sbert_batch_size,
                    "ce_batch_size": CONFIG.training.ce_batch_size,
                },
            )

            tracker = AccuracyTracker()
            tracker.record(entry)
            return entry

        except Exception as exc:
            logger.warning(f"Accuracy evaluation failed ({run_type}): {exc}", exc_info=True)
            return None

"""
Model Accuracy Tracker — Log, compare, and monitor model quality over time.

Maintains a JSONL log file (logs/accuracy_history.jsonl) where each line records
one accuracy measurement. The training pipeline writes two entries per run
(pre_training + post_training), enabling automatic before/after comparison.

Key capabilities:
- Score test pairs with SBERT (cosine similarity) and Cross-Encoder (logit scores)
- Compute AUC-ROC, binary accuracy, score gap, and per-label averages
- Persist every measurement to append-only JSONL for full audit trail
- Generate human-readable comparison reports with deltas and verdicts
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import CONFIG
from core.models import TrainingPair

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# Data Model
# ═══════════════════════════════════════════════

class AccuracyEntry:
    """A single accuracy measurement snapshot."""

    def __init__(
        self,
        run_type: str,
        training_run_id: str,
        model_version: str,
        ce_metrics: Optional[Dict] = None,
        sbert_metrics: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
        model_paths: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
    ):
        self.entry_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.run_type = run_type
        self.training_run_id = training_run_id
        self.model_version = model_version
        self.ce_metrics = ce_metrics or {}
        self.sbert_metrics = sbert_metrics or {}
        self.dataset_info = dataset_info or {}
        self.model_paths = model_paths or {}
        self.training_config = training_config or {}

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "run_type": self.run_type,
            "training_run_id": self.training_run_id,
            "model_version": self.model_version,
            "ce_metrics": self.ce_metrics,
            "sbert_metrics": self.sbert_metrics,
            "dataset_info": self.dataset_info,
            "model_paths": self.model_paths,
            "training_config": self.training_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccuracyEntry":
        entry = cls(
            run_type=data["run_type"],
            training_run_id=data["training_run_id"],
            model_version=data["model_version"],
            ce_metrics=data.get("ce_metrics", {}),
            sbert_metrics=data.get("sbert_metrics", {}),
            dataset_info=data.get("dataset_info", {}),
            model_paths=data.get("model_paths", {}),
            training_config=data.get("training_config", {}),
        )
        entry.entry_id = data.get("entry_id", entry.entry_id)
        entry.timestamp = data.get("timestamp", entry.timestamp)
        return entry


# ═══════════════════════════════════════════════
# Test Set Scorer
# ═══════════════════════════════════════════════

class TestSetScorer:
    """Score held-out test pairs with SBERT and Cross-Encoder to measure quality."""

    @staticmethod
    def score_cross_encoder(
        model_name_or_path: str,
        test_pairs: List[TrainingPair],
        batch_size: int = 64,
    ) -> Dict:
        """
        Score test pairs with a cross-encoder and compute classification metrics.

        Returns dict with: positive_avg_score, negative_avg_score, score_gap,
        accuracy, auc_roc, total, positives, negatives.
        """
        if not test_pairs:
            return {}

        from sentence_transformers.cross_encoder import CrossEncoder

        logger.info(f"Scoring {len(test_pairs)} test pairs with CE: {model_name_or_path}")
        model = CrossEncoder(model_name_or_path, max_length=256)

        sentence_pairs = [
            (p.anchor_text, p.candidate_text) for p in test_pairs
        ]
        labels = np.array([p.label for p in test_pairs])

        raw_scores = model.predict(
            sentence_pairs, batch_size=batch_size, show_progress_bar=False,
        )
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        scores = _sigmoid(raw_scores)

        pos_mask = labels == 1.0
        neg_mask = labels == 0.0

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        pos_avg = float(pos_scores.mean()) if pos_scores.size else None
        neg_avg = float(neg_scores.mean()) if neg_scores.size else None
        score_gap = (pos_avg - neg_avg) if (pos_avg is not None and neg_avg is not None) else None

        predictions = (scores >= 0.5).astype(int)
        accuracy = float((predictions == labels).mean())

        auc = _safe_auc(labels, scores)

        return {
            "positive_avg_score": _round(pos_avg),
            "negative_avg_score": _round(neg_avg),
            "score_gap": _round(score_gap),
            "accuracy": _round(accuracy),
            "auc_roc": _round(auc),
            "total": len(test_pairs),
            "positives": int(pos_mask.sum()),
            "negatives": int(neg_mask.sum()),
        }

    @staticmethod
    def score_sbert(
        model_name_or_path: str,
        test_pairs: List[TrainingPair],
        batch_size: int = 128,
    ) -> Dict:
        """
        Compute cosine similarities for test pairs with SBERT.

        Returns dict with: positive_avg_similarity, negative_avg_similarity,
        similarity_gap.
        """
        if not test_pairs:
            return {}

        from sentence_transformers import SentenceTransformer

        logger.info(f"Scoring {len(test_pairs)} test pairs with SBERT: {model_name_or_path}")
        model = SentenceTransformer(model_name_or_path)

        anchors = [p.anchor_text for p in test_pairs]
        candidates = [p.candidate_text for p in test_pairs]
        labels = np.array([p.label for p in test_pairs])

        a_emb = model.encode(anchors, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        c_emb = model.encode(candidates, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

        a_emb = a_emb.astype(np.float32)
        c_emb = c_emb.astype(np.float32)

        norms_a = np.linalg.norm(a_emb, axis=1, keepdims=True).clip(min=1e-8)
        norms_c = np.linalg.norm(c_emb, axis=1, keepdims=True).clip(min=1e-8)
        sims = np.sum((a_emb / norms_a) * (c_emb / norms_c), axis=1)

        pos_sims = sims[labels == 1.0]
        neg_sims = sims[labels == 0.0]

        pos_avg = float(pos_sims.mean()) if pos_sims.size else None
        neg_avg = float(neg_sims.mean()) if neg_sims.size else None
        sim_gap = (pos_avg - neg_avg) if (pos_avg is not None and neg_avg is not None) else None

        return {
            "positive_avg_similarity": _round(pos_avg),
            "negative_avg_similarity": _round(neg_avg),
            "similarity_gap": _round(sim_gap),
        }


# ═══════════════════════════════════════════════
# Accuracy Tracker (log persistence)
# ═══════════════════════════════════════════════

class AccuracyTracker:
    """
    Append-only JSONL logger for model accuracy over time.

    Each line in the log file is one AccuracyEntry in JSON.  Training scripts
    write two entries per run (pre_training + post_training) linked by
    ``training_run_id``.
    """

    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.path.join(CONFIG.logs_dir, "accuracy_history.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record(self, entry: AccuracyEntry) -> None:
        """Append one entry to the JSONL log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        logger.info(
            f"Accuracy entry recorded: run_type={entry.run_type}, "
            f"run_id={entry.training_run_id}"
        )

    def get_history(self) -> List[AccuracyEntry]:
        """Read all entries from the log."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(AccuracyEntry.from_dict(json.loads(line)))
        return entries

    def get_latest(self, run_type: str = None) -> Optional[AccuracyEntry]:
        """Get the most recent entry, optionally filtered by run_type."""
        entries = self.get_history()
        if run_type:
            entries = [e for e in entries if e.run_type == run_type]
        return entries[-1] if entries else None

    def get_run_pair(self, training_run_id: str) -> Tuple[Optional[AccuracyEntry], Optional[AccuracyEntry]]:
        """Get the (pre_training, post_training) entries for a given run."""
        entries = self.get_history()
        pre = None
        post = None
        for e in entries:
            if e.training_run_id == training_run_id:
                if e.run_type == "pre_training":
                    pre = e
                elif e.run_type == "post_training":
                    post = e
        return pre, post

    def get_latest_post_training(self) -> Optional[AccuracyEntry]:
        """Get the most recent post_training entry (previous best)."""
        return self.get_latest(run_type="post_training")


# ═══════════════════════════════════════════════
# Comparison Report
# ═══════════════════════════════════════════════

class ComparisonReport:
    """Generate human-readable accuracy comparison between two measurements."""

    @staticmethod
    def generate(
        before: AccuracyEntry,
        after: AccuracyEntry,
        output_path: str = None,
    ) -> str:
        """
        Build a formatted comparison report showing deltas for every metric.
        Saves both .txt and .json versions when output_path is given.
        """
        lines = [
            "",
            "=" * 72,
            "  MODEL ACCURACY COMPARISON",
            "=" * 72,
            f"  Training Run:  {after.training_run_id}",
            f"  Before Model:  {before.model_version}",
            f"  After Model:   {after.model_version}",
            f"  Compared At:   {datetime.utcnow().isoformat()}",
            "=" * 72,
        ]

        # Cross-Encoder metrics
        if before.ce_metrics and after.ce_metrics:
            lines.extend(_format_metric_section(
                "CROSS-ENCODER METRICS",
                before.ce_metrics,
                after.ce_metrics,
                metric_defs=[
                    ("positive_avg_score", "Positive Avg Score", True),
                    ("negative_avg_score", "Negative Avg Score", False),
                    ("score_gap", "Score Gap (pos - neg)", True),
                    ("accuracy", "Binary Accuracy", True),
                    ("auc_roc", "AUC-ROC", True),
                ],
            ))

        # SBERT metrics
        if before.sbert_metrics and after.sbert_metrics:
            lines.extend(_format_metric_section(
                "SBERT EMBEDDING METRICS",
                before.sbert_metrics,
                after.sbert_metrics,
                metric_defs=[
                    ("positive_avg_similarity", "Positive Avg Similarity", True),
                    ("negative_avg_similarity", "Negative Avg Similarity", False),
                    ("similarity_gap", "Similarity Gap", True),
                ],
            ))

        # Dataset info
        ds_before = before.dataset_info
        ds_after = after.dataset_info
        if ds_before or ds_after:
            lines.extend([
                "",
                f"  {'Dataset':<30} {'Before':>10} {'After':>10}",
                "  " + "-" * 55,
            ])
            for key in ("test_total", "test_positives", "test_negatives",
                        "train_total", "train_positives", "train_negatives"):
                b_val = ds_before.get(key, "-")
                a_val = ds_after.get(key, "-")
                lines.append(f"  {key:<30} {str(b_val):>10} {str(a_val):>10}")

        # Verdict
        verdict, details = _compute_verdict(before, after)
        lines.extend([
            "",
            "=" * 72,
            f"  VERDICT: {verdict}",
            "=" * 72,
        ])
        for d in details:
            lines.append(f"    {d}")
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            json_path = output_path.replace(".txt", ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "training_run_id": after.training_run_id,
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    "verdict": verdict,
                    "details": details,
                }, f, indent=2, default=str)
            logger.info(f"Comparison report saved: {output_path}")

        return report

    @staticmethod
    def generate_history_summary(tracker: AccuracyTracker) -> str:
        """Summarise the full accuracy history as a table."""
        entries = tracker.get_history()
        if not entries:
            return "No accuracy history found."

        lines = [
            "",
            "=" * 100,
            "  MODEL ACCURACY HISTORY",
            "=" * 100,
            "",
            f"  {'Timestamp':<22} {'Run Type':<16} {'Run ID':<24} "
            f"{'CE Acc':>8} {'CE AUC':>8} {'CE Gap':>8} {'SBERT Gap':>10}",
            "  " + "-" * 96,
        ]

        for e in entries:
            ce_acc = e.ce_metrics.get("accuracy", "-")
            ce_auc = e.ce_metrics.get("auc_roc", "-")
            ce_gap = e.ce_metrics.get("score_gap", "-")
            sb_gap = e.sbert_metrics.get("similarity_gap", "-")

            ce_acc_s = f"{ce_acc:.4f}" if isinstance(ce_acc, (int, float)) else str(ce_acc)
            ce_auc_s = f"{ce_auc:.4f}" if isinstance(ce_auc, (int, float)) else str(ce_auc)
            ce_gap_s = f"{ce_gap:.4f}" if isinstance(ce_gap, (int, float)) else str(ce_gap)
            sb_gap_s = f"{sb_gap:.4f}" if isinstance(sb_gap, (int, float)) else str(sb_gap)

            ts = e.timestamp[:19] if len(e.timestamp) > 19 else e.timestamp
            run_id_short = e.training_run_id[:22] if len(e.training_run_id) > 22 else e.training_run_id

            lines.append(
                f"  {ts:<22} {e.run_type:<16} {run_id_short:<24} "
                f"{ce_acc_s:>8} {ce_auc_s:>8} {ce_gap_s:>8} {sb_gap_s:>10}"
            )

        lines.extend(["", "=" * 100, ""])
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _round(val, decimals=6):
    if val is None:
        return None
    return round(float(val), decimals)


def _format_metric_section(
    title: str,
    before: dict,
    after: dict,
    metric_defs: List[Tuple[str, str, bool]],
) -> List[str]:
    """Format one section of the comparison report.

    metric_defs: list of (key, display_name, higher_is_better)
    """
    lines = [
        "",
        f"  -- {title} " + "-" * (60 - len(title)),
        "",
        f"  {'Metric':<32} {'Before':>10} {'After':>10} {'Delta':>10} {''}",
        "  " + "-" * 68,
    ]

    for key, display, higher_better in metric_defs:
        b_val = before.get(key)
        a_val = after.get(key)

        b_str = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else "   N/A"
        a_str = f"{a_val:.4f}" if isinstance(a_val, (int, float)) else "   N/A"

        if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)):
            delta = a_val - b_val
            arrow = "+" if delta > 0 else "" if delta < 0 else " "
            improved = (delta > 0) == higher_better if delta != 0 else True
            indicator = "GOOD" if improved else "REGR"
            delta_str = f"{arrow}{delta:.4f} {indicator}"
        else:
            delta_str = "       -"

        lines.append(f"  {display:<32} {b_str:>10} {a_str:>10}  {delta_str}")

    return lines


def _compute_verdict(before: AccuracyEntry, after: AccuracyEntry) -> Tuple[str, List[str]]:
    """Determine if the model improved, regressed, or is mixed."""
    improvements = []
    regressions = []

    ce_b = before.ce_metrics
    ce_a = after.ce_metrics

    if ce_b and ce_a:
        if _improved(ce_b, ce_a, "score_gap", higher_better=True):
            improvements.append("Cross-encoder score gap widened (better positive/negative separation)")
        elif _regressed(ce_b, ce_a, "score_gap", higher_better=True):
            regressions.append("Cross-encoder score gap narrowed")

        if _improved(ce_b, ce_a, "auc_roc", higher_better=True):
            improvements.append("Cross-encoder AUC-ROC improved")
        elif _regressed(ce_b, ce_a, "auc_roc", higher_better=True):
            regressions.append("Cross-encoder AUC-ROC declined")

        if _improved(ce_b, ce_a, "accuracy", higher_better=True):
            improvements.append("Cross-encoder binary accuracy improved")
        elif _regressed(ce_b, ce_a, "accuracy", higher_better=True):
            regressions.append("Cross-encoder binary accuracy declined")

        if _improved(ce_b, ce_a, "positive_avg_score", higher_better=True):
            improvements.append("Positive match scores increased")

        if _improved(ce_b, ce_a, "negative_avg_score", higher_better=False):
            improvements.append("Negative match scores decreased")

    sb_b = before.sbert_metrics
    sb_a = after.sbert_metrics

    if sb_b and sb_a:
        if _improved(sb_b, sb_a, "similarity_gap", higher_better=True):
            improvements.append("SBERT similarity gap widened")
        elif _regressed(sb_b, sb_a, "similarity_gap", higher_better=True):
            regressions.append("SBERT similarity gap narrowed")

    details = []
    if improvements:
        details.append("Improvements:")
        for imp in improvements:
            details.append(f"  + {imp}")
    if regressions:
        details.append("Regressions:")
        for reg in regressions:
            details.append(f"  - {reg}")

    if not regressions and improvements:
        return "IMPROVED", details
    elif regressions and not improvements:
        return "REGRESSED", details
    elif regressions and improvements:
        return "MIXED", details
    else:
        return "NO CHANGE", ["No significant metric changes detected."]


def _improved(before: dict, after: dict, key: str, higher_better: bool) -> bool:
    b = before.get(key)
    a = after.get(key)
    if b is None or a is None:
        return False
    threshold = 0.001
    return (a - b > threshold) if higher_better else (b - a > threshold)


def _regressed(before: dict, after: dict, key: str, higher_better: bool) -> bool:
    b = before.get(key)
    a = after.get(key)
    if b is None or a is None:
        return False
    threshold = 0.001
    return (b - a > threshold) if higher_better else (a - b > threshold)


# ═══════════════════════════════════════════════
# Convenience: resolve current model paths
# ═══════════════════════════════════════════════

def get_current_model_paths() -> Dict[str, str]:
    """Return the model paths currently configured for production use."""
    sbert = (
        CONFIG.model.tuned_sbert_path
        if CONFIG.model.use_tuned_sbert and CONFIG.model.tuned_sbert_path
        else CONFIG.model.sbert_model
    )
    ce = (
        CONFIG.model.tuned_cross_encoder_path
        if CONFIG.model.use_tuned_cross_encoder and CONFIG.model.tuned_cross_encoder_path
        else CONFIG.model.cross_encoder_model
    )
    return {"sbert": sbert, "cross_encoder": ce}

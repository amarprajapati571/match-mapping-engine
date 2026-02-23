"""
Standalone Accuracy Evaluation — Score models on labeled data and track improvement.

Run this at any time to measure how well the current (or custom) models perform
on a set of labeled training pairs, and compare against previous measurements.

Usage:
    # Evaluate current models on labeled data
    python scripts/evaluate_accuracy.py --data-file data/training_pairs.json

    # Evaluate specific tuned models
    python scripts/evaluate_accuracy.py --data-file data/training_pairs.json \
        --sbert-path models/sbert_tuned_20260220 \
        --ce-path models/ce_tuned_20260220

    # View full accuracy history
    python scripts/evaluate_accuracy.py --history

    # Compare two specific training runs
    python scripts/evaluate_accuracy.py --compare run_20260220_140000
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.models import TrainingPair
from evaluation.accuracy_tracker import (
    AccuracyEntry,
    AccuracyTracker,
    ComparisonReport,
    TestSetScorer,
    get_current_model_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_accuracy")


def load_test_pairs(data_file: str) -> list:
    """Load TrainingPair objects from a JSON file."""
    with open(data_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    pairs = []
    for rec in records:
        if isinstance(rec, dict) and "anchor_text" in rec and "candidate_text" in rec:
            pairs.append(TrainingPair(**rec))
        else:
            logger.warning(f"Skipping invalid record: {str(rec)[:80]}...")

    logger.info(f"Loaded {len(pairs)} test pairs from {data_file}")
    return pairs


def run_evaluation(pairs: list, sbert_path: str, ce_path: str, run_id: str, version: str):
    """Score pairs, record to log, and print results."""
    tracker = AccuracyTracker()

    print(f"\n  Evaluating {len(pairs)} pairs...")
    print(f"  SBERT:         {sbert_path}")
    print(f"  Cross-Encoder: {ce_path}\n")

    ce_metrics = TestSetScorer.score_cross_encoder(ce_path, pairs)
    sbert_metrics = TestSetScorer.score_sbert(sbert_path, pairs)

    pos_count = sum(1 for p in pairs if p.label == 1.0)
    neg_count = sum(1 for p in pairs if p.label == 0.0)

    entry = AccuracyEntry(
        run_type="evaluation",
        training_run_id=run_id,
        model_version=version,
        ce_metrics=ce_metrics,
        sbert_metrics=sbert_metrics,
        dataset_info={
            "test_total": len(pairs),
            "test_positives": pos_count,
            "test_negatives": neg_count,
        },
        model_paths={"sbert": sbert_path, "cross_encoder": ce_path},
    )

    tracker.record(entry)

    # Print results
    print("=" * 60)
    print("  ACCURACY EVALUATION RESULTS")
    print("=" * 60)

    if ce_metrics:
        print("\n  Cross-Encoder Metrics:")
        print(f"    Positive Avg Score:    {ce_metrics.get('positive_avg_score', 'N/A')}")
        print(f"    Negative Avg Score:    {ce_metrics.get('negative_avg_score', 'N/A')}")
        print(f"    Score Gap:             {ce_metrics.get('score_gap', 'N/A')}")
        print(f"    Binary Accuracy:       {ce_metrics.get('accuracy', 'N/A')}")
        print(f"    AUC-ROC:               {ce_metrics.get('auc_roc', 'N/A')}")

    if sbert_metrics:
        print("\n  SBERT Metrics:")
        print(f"    Positive Avg Sim:      {sbert_metrics.get('positive_avg_similarity', 'N/A')}")
        print(f"    Negative Avg Sim:      {sbert_metrics.get('negative_avg_similarity', 'N/A')}")
        print(f"    Similarity Gap:        {sbert_metrics.get('similarity_gap', 'N/A')}")

    print(f"\n  Dataset: {len(pairs)} pairs ({pos_count} pos, {neg_count} neg)")
    print(f"  Logged to: {tracker.log_path}")

    # Auto-compare with previous post_training if available
    prev = tracker.get_latest_post_training()
    if prev and prev.entry_id != entry.entry_id:
        print("\n  Comparing with most recent post-training baseline...")
        report = ComparisonReport.generate(prev, entry)
        print(report)

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy and track improvement over time"
    )
    parser.add_argument("--data-file", help="Path to training_pairs.json or labeled data")
    parser.add_argument("--sbert-path", default=None, help="Path to SBERT model (default: current config)")
    parser.add_argument("--ce-path", default=None, help="Path to Cross-Encoder model (default: current config)")
    parser.add_argument("--version", default=None, help="Model version label for this evaluation")
    parser.add_argument("--history", action="store_true", help="Print full accuracy history and exit")
    parser.add_argument("--compare", default=None, metavar="RUN_ID",
                        help="Compare pre/post for a specific training run ID")
    args = parser.parse_args()

    tracker = AccuracyTracker()

    if args.history:
        print(ComparisonReport.generate_history_summary(tracker))
        return

    if args.compare:
        pre, post = tracker.get_run_pair(args.compare)
        if not pre or not post:
            found = "pre only" if pre else ("post only" if post else "neither")
            print(f"  Could not find pre+post pair for run '{args.compare}' ({found})")
            entries = tracker.get_history()
            matching = [e for e in entries if e.training_run_id == args.compare]
            if matching:
                print(f"  Found {len(matching)} entries for this run:")
                for e in matching:
                    print(f"    - {e.run_type} at {e.timestamp}")
            return
        report = ComparisonReport.generate(pre, post)
        print(report)
        return

    if not args.data_file:
        parser.error("--data-file is required for evaluation (or use --history / --compare)")

    pairs = load_test_pairs(args.data_file)
    if not pairs:
        print("  No valid test pairs found. Check the data file format.")
        return

    current_paths = get_current_model_paths()
    sbert_path = args.sbert_path or current_paths["sbert"]
    ce_path = args.ce_path or current_paths["cross_encoder"]

    run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    version = args.version or f"eval_{datetime.utcnow().strftime('%Y%m%d')}"

    run_evaluation(pairs, sbert_path, ce_path, run_id, version)

    print(ComparisonReport.generate_history_summary(tracker))


if __name__ == "__main__":
    main()

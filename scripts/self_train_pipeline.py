"""
Self-Training Pipeline — Fetch CSE feedback and retrain models.

Fetches human feedback from the CSE team's review API, converts it into
training pairs, fine-tunes SBERT + Cross-Encoder, and optionally reloads
the models into the running inference engine.

Feedback types and their training effect:
  Correct      → positive pair   → model learns to score these HIGHER
  Not correct  → hard negative   → model learns to score these LOWER
  Need to swap → positive (swap) → model learns correct team ordering
  Not Sure     → skipped         → no training signal from uncertain data

Usage:
    python scripts/self_train_pipeline.py --platform ODDSPORTAL
    python scripts/self_train_pipeline.py --platform ODDSPORTAL --use-local
    python scripts/self_train_pipeline.py --platform ODDSPORTAL --sbert-only
    python scripts/self_train_pipeline.py --platform ODDSPORTAL --ce-only
    python scripts/self_train_pipeline.py --platform ODDSPORTAL --dry-run
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.feedback import FeedbackStore, CSEFeedbackLoader
from training.trainer import TrainingOrchestrator, DatasetBuilder
from evaluation.accuracy_tracker import AccuracyTracker, ComparisonReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("self_train")


def save_feedback_snapshot(feedback_rows: list, output_path: str):
    """Save raw feedback data for audit trail."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback_rows, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Feedback snapshot saved → {output_path}")


def save_training_report(report: dict, output_path: str):
    """Save training report for tracking model versions."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Training report saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Self-train models from CSE team feedback"
    )
    parser.add_argument(
        "--platform", default="ODDSPORTAL",
        help="Platform to fetch feedback for (default: ODDSPORTAL)",
    )
    parser.add_argument(
        "--use-local", action="store_true",
        help="Use local API (localhost:8010) instead of production",
    )
    parser.add_argument(
        "--api-url", default=None,
        help="Override feedback API URL entirely",
    )
    parser.add_argument(
        "--sbert-only", action="store_true",
        help="Only train SBERT bi-encoder",
    )
    parser.add_argument(
        "--ce-only", action="store_true",
        help="Only train Cross-Encoder reranker",
    )
    parser.add_argument(
        "--sbert-output", default=None,
        help="Custom output path for tuned SBERT model",
    )
    parser.add_argument(
        "--ce-output", default=None,
        help="Custom output path for tuned Cross-Encoder model",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--input-file", default=None,
        help="Use cached CSE feedback JSON file instead of fetching from API",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch feedback and build pairs but don't train",
    )
    parser.add_argument(
        "--include-existing", default=None,
        help="Path to existing labeled_records.json to merge with CSE feedback",
    )
    parser.add_argument(
        "--min-feedback", type=int, default=None,
        help="Minimum feedback rows required to start training",
    )
    args = parser.parse_args()

    if args.epochs:
        CONFIG.training.sbert_epochs = args.epochs
        CONFIG.training.ce_epochs = args.epochs
    if args.batch_size:
        CONFIG.training.sbert_batch_size = args.batch_size
        CONFIG.training.ce_batch_size = args.batch_size
    if args.use_local:
        CONFIG.feedback_api.use_local = True

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    min_feedback = args.min_feedback or CONFIG.feedback_api.min_feedback_for_training

    print("\n" + "=" * 70)
    print("  SELF-TRAINING PIPELINE — CSE FEEDBACK → MODEL IMPROVEMENT")
    print("=" * 70 + "\n")

    # ── Step 1: Fetch / Load CSE Feedback ──
    if args.input_file:
        print(f"Step 1: Loading cached CSE feedback from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            feedback_rows = json.load(f)
        if not feedback_rows:
            print("\n  Cached file is empty. Nothing to train on.")
            return
        print(f"  Loaded {len(feedback_rows)} feedback rows from file")
        snapshot_path = args.input_file
    else:
        print("Step 1: Fetching CSE team feedback...")
        feedback_rows = CSEFeedbackLoader.fetch_feedback(
            platform=args.platform,
            url=args.api_url,
        )
        if not feedback_rows:
            print("\n  No feedback rows returned from API. Nothing to train on.")
            return
        print(f"  Fetched {len(feedback_rows)} feedback rows")
        os.makedirs("data", exist_ok=True)
        snapshot_path = os.path.join("data", f"cse_feedback_{timestamp}.json")
        save_feedback_snapshot(feedback_rows, snapshot_path)

    # ── Step 2: Convert Feedback to Training Pairs ──
    print("\nStep 2: Converting CSE feedback to training pairs...")
    store = FeedbackStore()

    if args.include_existing:
        print(f"  Merging with existing labels from {args.include_existing}...")
        from core.feedback import BulkFeedbackLoader
        with open(args.include_existing) as f:
            existing_records = json.load(f)
        existing_pairs = BulkFeedbackLoader.load_from_records(existing_records, store)
        print(f"  Loaded {existing_pairs} pairs from existing labels")

    n_pairs, feedback_counts = CSEFeedbackLoader.convert_to_training_pairs(
        feedback_rows, store
    )

    print(f"\n  Feedback breakdown:")
    print(f"    Correct (→ positive):     {feedback_counts['correct']}")
    print(f"    Not correct (→ negative): {feedback_counts['not_correct']}")
    print(f"    Need to swap (→ swap):    {feedback_counts['need_to_swap']}")
    print(f"    Not Sure (skipped):       {feedback_counts['not_sure_skipped']}")
    print(f"    Errors:                   {feedback_counts['errors']}")
    print(f"    Total training pairs:     {n_pairs}")

    all_pairs = store.get_training_pairs()
    positives = store.get_positives()
    negatives = store.get_hard_negatives()

    print(f"\n  Combined training data:")
    print(f"    Total pairs:     {len(all_pairs)}")
    print(f"    Positives:       {len(positives)}")
    print(f"    Hard negatives:  {len(negatives)}")

    if not positives and negatives:
        print("\n  WARNING: Only NEGATIVE examples found (no 'Correct' feedback).")
        print("  The system will attempt to load cached positives from training_pairs.json")
        print("  or generate synthetic data to enable proper two-class training.")
    elif positives and not negatives:
        print("\n  WARNING: Only POSITIVE examples found (no 'Not correct' feedback).")
        print("  Hard negatives will be generated automatically by shuffling candidates.")
    elif positives and negatives:
        ratio = min(len(positives), len(negatives)) / max(len(positives), len(negatives))
        print(f"    Class ratio:     {ratio:.2f} (min/max)")
        if ratio < 0.1:
            print("  WARNING: Severe class imbalance. Auto-balancing will be applied.")

    trainable_feedback = (
        feedback_counts["correct"]
        + feedback_counts["not_correct"]
        + feedback_counts["need_to_swap"]
    )
    if trainable_feedback < min_feedback:
        print(
            f"\n  Insufficient trainable feedback ({trainable_feedback} < {min_feedback}). "
            f"Collect more CSE reviews before training."
        )
        print("  Use --min-feedback to override this threshold.")
        return

    # ── Step 3: Validate ──
    print("\nStep 3: Validating training data...")
    if not DatasetBuilder.validate_no_contamination(all_pairs):
        print("  ABORTING: Label contamination detected!")
        sys.exit(1)
    print("  Data validation passed.")

    if args.dry_run:
        print("\n  --dry-run: Skipping training. Data looks good!")
        print(f"  Feedback snapshot: {snapshot_path}")
        return

    # ── Step 4: Train Models ──
    print("\nStep 4: Training models from CSE feedback...")
    os.makedirs("models", exist_ok=True)

    sbert_out = args.sbert_output or f"models/sbert_cse_tuned_{timestamp}"
    ce_out = args.ce_output or f"models/ce_cse_tuned_{timestamp}"

    orchestrator = TrainingOrchestrator(all_pairs)
    stats = orchestrator.prepare()

    training_result = {
        "timestamp": timestamp,
        "platform": args.platform,
        "feedback_counts": feedback_counts,
        "total_feedback_rows": len(feedback_rows),
        "total_training_pairs": len(all_pairs),
        "dataset_stats": stats,
    }

    if args.ce_only:
        print("  Training Cross-Encoder only...")
        ce_path = orchestrator.train_cross_encoder(ce_out)
        if ce_path:
            print(f"  Cross-Encoder saved: {ce_path}")
        else:
            print("  Cross-Encoder training skipped — no examples.")
        training_result["cross_encoder_path"] = ce_path
    elif args.sbert_only:
        print("  Training SBERT only...")
        sbert_path = orchestrator.train_sbert(sbert_out)
        if sbert_path:
            print(f"  SBERT saved: {sbert_path}")
        else:
            print("  SBERT training skipped — no data available.")
        training_result["sbert_path"] = sbert_path
    else:
        print("  Training both SBERT + Cross-Encoder...")
        result = orchestrator.train_all(sbert_out, ce_out, track_accuracy=True)
        training_result.update(result)

        sbert_out = result.get("sbert_model_path") or "SKIPPED (no data)"
        ce_out = result.get("cross_encoder_model_path") or "SKIPPED (no data)"

        if result.get("accuracy_comparison"):
            print(result["accuracy_comparison"])

    # ── Step 5: Save Report ──
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"self_train_report_{timestamp}.json")
    save_training_report(training_result, report_path)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SELF-TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
  CSE Feedback:
    Total rows fetched:  {len(feedback_rows)}
    Correct:             {feedback_counts['correct']}
    Not correct:         {feedback_counts['not_correct']}
    Need to swap:        {feedback_counts['need_to_swap']}
    Not Sure (skipped):  {feedback_counts['not_sure_skipped']}

  Training:
    Total training pairs: {len(all_pairs)}
    Positives:            {len(positives)}
    Hard negatives:       {len(negatives)}

  Expected score changes after training:
    Previously correct (e.g. 9.0)   → should now score HIGHER (e.g. 9.5)
    Previously incorrect (e.g. 8.5) → should now score LOWER  (e.g. 7.0)

  Model artifacts:
    SBERT:        {sbert_out if not args.ce_only else 'N/A'}
    Cross-Encoder: {ce_out if not args.sbert_only else 'N/A'}
    Report:       {report_path}
    Feedback:     {snapshot_path}

  To deploy the new models:
    POST /models/reload with the paths above, or run:
    curl -X POST http://localhost:8000/models/reload \\
      -H "Content-Type: application/json" \\
      -d '{{"use_tuned_sbert": true, "use_tuned_cross_encoder": true, \\
           "tuned_sbert_path": "{sbert_out}", \\
           "tuned_cross_encoder_path": "{ce_out}"}}'
""")

    tracker = AccuracyTracker()
    print(ComparisonReport.generate_history_summary(tracker))
    print(f"  Accuracy log: {os.path.join(CONFIG.logs_dir, 'accuracy_history.jsonl')}")


if __name__ == "__main__":
    main()

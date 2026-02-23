"""
Training Script — Fine-tune SBERT + CrossEncoder from human labels.

Usage:
    python scripts/train_models.py --data-file data/labeled_records.json
    python scripts/train_models.py --data-file data/labeled_records.json --sbert-only
    python scripts/train_models.py --data-file data/labeled_records.json --ce-only
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.feedback import FeedbackStore, BulkFeedbackLoader
from training.trainer import TrainingOrchestrator, DatasetBuilder
from evaluation.accuracy_tracker import AccuracyTracker, ComparisonReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train match mapping models")
    parser.add_argument("--data-file", required=True, help="Path to labeled records JSON")
    parser.add_argument("--sbert-only", action="store_true", help="Only train SBERT")
    parser.add_argument("--ce-only", action="store_true", help="Only train CrossEncoder")
    parser.add_argument("--sbert-output", default=None, help="SBERT output path")
    parser.add_argument("--ce-output", default=None, help="CrossEncoder output path")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--no-accuracy-tracking", action="store_true",
                        help="Skip pre/post accuracy evaluation")
    args = parser.parse_args()
    
    # Override config if args provided
    if args.epochs:
        CONFIG.training.sbert_epochs = args.epochs
        CONFIG.training.ce_epochs = args.epochs
    if args.batch_size:
        CONFIG.training.sbert_batch_size = args.batch_size
        CONFIG.training.ce_batch_size = args.batch_size
    
    # Load data
    logger.info(f"Loading labeled records from {args.data_file}")
    with open(args.data_file) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} records")
    
    # Build training pairs
    store = FeedbackStore()
    n_pairs = BulkFeedbackLoader.load_from_records(records, store)
    pairs = store.get_training_pairs()
    
    logger.info(f"Generated {n_pairs} training pairs")
    logger.info(f"  Positives:      {len(store.get_positives())}")
    logger.info(f"  Hard negatives: {len(store.get_hard_negatives())}")
    
    # Validate
    if not DatasetBuilder.validate_no_contamination(pairs):
        logger.error("ABORTING: Label contamination detected!")
        sys.exit(1)
    
    # Train
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    
    sbert_out = args.sbert_output or f"models/sbert_tuned_{timestamp}"
    ce_out = args.ce_output or f"models/ce_tuned_{timestamp}"
    
    orchestrator = TrainingOrchestrator(pairs)
    track = not args.no_accuracy_tracking

    if args.ce_only:
        orchestrator.prepare()
        logger.info("Training CrossEncoder only...")
        ce_path = orchestrator.train_cross_encoder(ce_out)
        logger.info(f"CrossEncoder saved: {ce_path}")
    elif args.sbert_only:
        orchestrator.prepare()
        logger.info("Training SBERT only...")
        sbert_path = orchestrator.train_sbert(sbert_out)
        logger.info(f"SBERT saved: {sbert_path}")
    else:
        logger.info("Training both models...")
        result = orchestrator.train_all(sbert_out, ce_out, track_accuracy=track)
        logger.info(f"Training complete: {json.dumps(result, indent=2, default=str)}")

        if result.get("accuracy_comparison"):
            print(result["accuracy_comparison"])

    # Print accuracy history summary
    if track:
        tracker = AccuracyTracker()
        print(ComparisonReport.generate_history_summary(tracker))

    print("\n✓ Training complete!")
    print(f"  To deploy, call POST /models/reload with the new model paths.")
    print(f"  SBERT:        {sbert_out}")
    print(f"  CrossEncoder: {ce_out}")
    print(f"\n  Accuracy log: {os.path.join(CONFIG.logs_dir, 'accuracy_history.jsonl')}")


if __name__ == "__main__":
    main()

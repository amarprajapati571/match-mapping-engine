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
    python scripts/self_train_pipeline.py --all-platforms
    python scripts/self_train_pipeline.py --all-platforms --dry-run
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
        "--all-platforms", action="store_true",
        help="Fetch from ALL platforms (ODDSPORTAL, SBO, FLASHSCORE, SOFASCORE) "
             "and train on combined dataset. Overrides --platform.",
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
    parser.add_argument(
        "--include-aliases", action="store_true",
        help="Merge alias training data (from admin_leagues.json + admin_teams.json) "
             "with CSE feedback for a larger, more balanced dataset (~55K pairs)",
    )
    parser.add_argument(
        "--leagues-file", default="data/admin_leagues.json",
        help="Path to leagues JSON for --include-aliases (default: data/admin_leagues.json)",
    )
    parser.add_argument(
        "--teams-file", default="data/admin_teams.json",
        help="Path to teams JSON for --include-aliases (default: data/admin_teams.json)",
    )
    parser.add_argument(
        "--temporal-split", action="store_true",
        help="Use temporal train/test split (train on older data, test on newer) "
             "for more realistic evaluation",
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
        per_platform_counts = {"cached_file": len(feedback_rows)}
    else:
        # Determine which platforms to fetch
        if args.all_platforms:
            platforms = [
                p.strip().upper()
                for p in CONFIG.feedback_api.self_train_platforms.split(",")
                if p.strip()
            ]
            print(f"Step 1: Fetching CSE feedback from ALL platforms: {platforms}")
        else:
            platforms = [args.platform]
            print(f"Step 1: Fetching CSE team feedback from {args.platform}...")

        # Fetch feedback from each platform sequentially
        feedback_rows = []
        per_platform_counts = {}

        for platform in platforms:
            print(f"\n  Fetching from {platform}...")
            try:
                rows = CSEFeedbackLoader.fetch_feedback(
                    platform=platform,
                    url=args.api_url,
                )
                row_count = len(rows) if rows else 0
                per_platform_counts[platform] = row_count
                if rows:
                    feedback_rows.extend(rows)
                    print(f"    {platform}: {row_count} feedback rows")
                else:
                    print(f"    {platform}: no data returned")
            except Exception as e:
                logger.error(f"Failed to fetch from {platform}: {e}")
                per_platform_counts[platform] = 0
                print(f"    {platform}: ERROR - {e}")

        # Per-platform summary
        if args.all_platforms:
            active_platforms = sum(1 for c in per_platform_counts.values() if c > 0)
            print(f"\n  Per-platform summary:")
            for plat, count in per_platform_counts.items():
                status = f"{count} rows" if count > 0 else "no data"
                print(f"    {plat}: {status}")
            print(f"    TOTAL: {len(feedback_rows)} rows from {active_platforms}/{len(platforms)} platforms")

        if not feedback_rows:
            print("\n  No feedback rows returned from API. Nothing to train on.")
            return
        print(f"\n  Total fetched: {len(feedback_rows)} feedback rows")
        os.makedirs("data", exist_ok=True)
        if args.all_platforms:
            snapshot_path = os.path.join("data", f"cse_feedback_all_platforms_{timestamp}.json")
        else:
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

    # ── Merge Alias Training Data (optional) ──
    alias_pairs_count = 0
    if args.include_aliases:
        print(f"\n  Merging alias training data...")
        try:
            from scripts.train_from_aliases import (
                generate_league_pairs, generate_team_pairs, generate_hard_negatives,
            )
            from core.league_loader import load_leagues_from_file
            from core.team_loader import load_teams_from_file

            leagues = load_leagues_from_file(args.leagues_file)
            teams = load_teams_from_file(args.teams_file)

            if leagues or teams:
                league_pairs = generate_league_pairs(leagues) if leagues else []
                team_pairs = generate_team_pairs(teams) if teams else []
                alias_positives = league_pairs + team_pairs

                # Generate balanced hard negatives for alias data
                alias_negatives = generate_hard_negatives(
                    teams, n_negatives=len(alias_positives),
                ) if teams else []

                for p in alias_positives + alias_negatives:
                    store.add_training_pair(p)
                alias_pairs_count = len(alias_positives) + len(alias_negatives)

                print(f"    Alias league pairs:  {len(league_pairs)}")
                print(f"    Alias team pairs:    {len(team_pairs)}")
                print(f"    Alias negatives:     {len(alias_negatives)}")
                print(f"    Alias total:         {alias_pairs_count}")
            else:
                print(f"    WARNING: No leagues/teams data found at {args.leagues_file} / {args.teams_file}")
        except Exception as e:
            logger.warning(f"Failed to load alias data: {e}")
            print(f"    WARNING: Alias merge failed: {e}")

    print(f"\n  Feedback breakdown:")
    print(f"    Correct (→ positive):       {feedback_counts['correct']}")
    print(f"    Not correct (→ negative):   {feedback_counts['not_correct']}")
    print(f"    Need to swap (→ swap):      {feedback_counts['need_to_swap']}")
    print(f"    Not Sure (skipped):         {feedback_counts['not_sure_skipped']}")
    print(f"    No B365 data (skipped):     {feedback_counts.get('no_b365_skipped', 'N/A')}")
    print(f"    No feedback value:          {feedback_counts.get('no_feedback_value', 'N/A')}")
    print(f"    Unknown feedback values:    {feedback_counts.get('unknown_feedback_value', 'N/A')}")
    print(f"    Errors:                     {feedback_counts['errors']}")
    if feedback_counts.get("contamination_resolved"):
        print(f"    Contamination resolved:     {feedback_counts['contamination_resolved']} (duplicate pairs removed)")
    print(f"    Total training pairs:       {n_pairs}")

    # Show where the unaccounted rows went
    accounted = (
        feedback_counts["correct"] + feedback_counts["not_correct"]
        + feedback_counts["need_to_swap"] + feedback_counts["not_sure_skipped"]
        + feedback_counts.get("no_b365_skipped", 0)
        + feedback_counts.get("no_feedback_value", 0)
        + feedback_counts.get("unknown_feedback_value", 0)
        + feedback_counts["errors"]
    )
    unaccounted = len(feedback_rows) - accounted
    if unaccounted > 0:
        print(f"    Unaccounted rows:           {unaccounted}")
    print(f"    Total rows from API:        {len(feedback_rows)}")

    all_pairs = store.get_training_pairs()
    positives = store.get_positives()
    negatives = store.get_hard_negatives()

    print(f"\n  Combined training data:")
    print(f"    Total pairs:     {len(all_pairs)}")
    print(f"    Positives:       {len(positives)}")
    print(f"    Hard negatives:  {len(negatives)}")
    if alias_pairs_count > 0:
        print(f"    (includes {alias_pairs_count} alias pairs)")

    if not positives and negatives:
        print("\n  !! SINGLE-CLASS ALERT: Only NEGATIVE examples found (0 positives).")
        print("  This has been happening since Feb 23 — likely a feedback API issue.")
        print("  The system will attempt to load cached positives or generate synthetic data.")
    elif positives and not negatives:
        print("\n  WARNING: Only POSITIVE examples found (no 'Not correct' feedback).")
        print("  Hard negatives will be generated automatically by shuffling candidates.")
    elif positives and negatives:
        ratio = min(len(positives), len(negatives)) / max(len(positives), len(negatives))
        pos_neg_ratio = len(positives) / max(len(negatives), 1)
        print(f"    Class ratio:     {ratio:.2f} (min/max)")
        print(f"    Pos:Neg ratio:   {pos_neg_ratio:.2f}:1")
        if ratio < 0.1:
            print("  WARNING: Severe class imbalance. Auto-balancing will be applied.")
        if pos_neg_ratio > 3.0:
            print("  ⚠ RATIO DRIFT WARNING: Pos:Neg ratio exceeds 3:1.")
            print(f"    Optimal is ~1.4:1. Current {pos_neg_ratio:.2f}:1 may limit SBERT quality.")
            print(f"    Consider enriching hard negatives (same-league same-date mining).")
            logger.warning(
                f"Ratio drift: {pos_neg_ratio:.2f}:1 (optimal ~1.4:1). "
                f"{len(positives)} positives, {len(negatives)} negatives."
            )
        elif pos_neg_ratio < 0.33:
            print("  ⚠ RATIO WARNING: Too many negatives relative to positives.")
            logger.warning(f"Ratio {pos_neg_ratio:.2f}:1 — too negative-heavy.")

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

    # ── Validation Gate: Block single-class training ──
    if len(positives) == 0 and len(negatives) > 0:
        print("\n" + "!" * 70)
        print("  TRAINING BLOCKED — Zero positive examples detected!")
        print("!" * 70)
        print(f"""
  The CSE feedback contains {len(negatives)} negatives but 0 positives.
  Training on single-class data produces a useless model that always
  predicts 'no match'. This wastes compute and disk space.

  Diagnostics:
    Total feedback rows:    {len(feedback_rows)}
    Correct (→ positive):   {feedback_counts['correct']}
    Not correct (→ neg):    {feedback_counts['not_correct']}
    Need to swap:           {feedback_counts['need_to_swap']}
    Not Sure (skipped):     {feedback_counts['not_sure_skipped']}
    No B365 data (skipped): {feedback_counts.get('no_b365_skipped', 'N/A')}
    Errors:                 {feedback_counts['errors']}

  Possible causes:
    1. CSE team is not marking any matches as 'Correct' in the feedback UI
    2. The feedback API is not returning 'Correct' entries (query/filter bug)
    3. The feedback value extraction is not recognizing the 'Correct' format

  To investigate:
    - Run with --dry-run to inspect feedback data without training
    - Check a feedback snapshot: data/cse_feedback_*.json
    - Verify the CSE feedback API returns rows with 'Correct' status

  To override (NOT recommended):
    - Use --include-existing with a labeled_records.json containing positives
""")
        logger.error(
            f"TRAINING BLOCKED: 0 positives, {len(negatives)} negatives. "
            f"Feedback counts: {feedback_counts}"
        )
        return

    if len(negatives) == 0 and len(positives) > 0:
        print("\n  WARNING: No negatives found. Synthetic hard negatives will be generated.")
        print("  Training will proceed but may not be optimal.")

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

    orchestrator = TrainingOrchestrator(all_pairs, temporal_split=args.temporal_split)
    stats = orchestrator.prepare()

    training_result = {
        "timestamp": timestamp,
        "platform": "ALL" if args.all_platforms else args.platform,
        "platforms_fetched": per_platform_counts,
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

        # ── Post-training regression check (P0) ──
        ce_acc = result.get("ce_accuracy")
        sbert_gap = result.get("sbert_gap")
        prev_ce_acc = result.get("prev_ce_accuracy")
        prev_sbert_gap = result.get("prev_sbert_gap")

        if ce_acc is not None and prev_ce_acc is not None:
            ce_drop = prev_ce_acc - ce_acc
            if ce_drop > 0.01:
                print(f"\n  ⚠ REGRESSION WARNING: CE accuracy dropped {ce_drop:.4f} "
                      f"({prev_ce_acc:.4f} → {ce_acc:.4f})")
                logger.warning(
                    f"Post-training regression: CE accuracy dropped {ce_drop:.4f} "
                    f"({prev_ce_acc:.4f} → {ce_acc:.4f}). "
                    f"Consider reverting to previous model."
                )

        if sbert_gap is not None and prev_sbert_gap is not None:
            gap_drop = prev_sbert_gap - sbert_gap
            if gap_drop > 0.05:
                print(f"\n  ⚠ REGRESSION WARNING: SBERT gap dropped {gap_drop:.4f} "
                      f"({prev_sbert_gap:.4f} → {sbert_gap:.4f})")
                logger.warning(
                    f"Post-training regression: SBERT gap dropped {gap_drop:.4f} "
                    f"({prev_sbert_gap:.4f} → {sbert_gap:.4f}). "
                    f"Consider reverting to previous model."
                )

    # ── Step 5: Save Report ──
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"self_train_report_{timestamp}.json")
    save_training_report(training_result, report_path)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SELF-TRAINING COMPLETE")
    print("=" * 70)
    # Show per-platform breakdown in summary
    if args.all_platforms:
        print(f"\n  Platforms fetched:")
        for plat, count in per_platform_counts.items():
            status = f"{count} rows" if count > 0 else "no data"
            print(f"    {plat}: {status}")

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

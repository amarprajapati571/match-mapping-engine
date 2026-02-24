"""
Automated Scheduler — Runs the full pipeline on a configurable interval.

Cycle (repeats every SCHEDULER_INTERVAL_MINUTES, default 45):

  Phase 1  SELF-TRAIN
    Fetch CSE feedback → convert to training pairs → train models → reload

  Phase 2  FETCH DATA
    Fetch all provider matches concurrently — each API returns provider + Bet365 data

  Phase 3  INFERENCE
    Per-provider: index that provider's Bet365 pool → batch inference → save → optionally push

All timings are configurable via environment variables (see .env.example).

Usage:
    python scripts/scheduler.py                          # run with defaults
    SCHEDULER_INTERVAL_MINUTES=30 python scripts/scheduler.py
    python scripts/scheduler.py --run-once               # single run, no loop
    python scripts/scheduler.py --skip-training           # skip Phase 1
    python scripts/scheduler.py --skip-inference          # skip Phase 2+3
"""

import sys
import os
import time
import signal
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Event
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.inference import InferenceEngine
from core.feedback import FeedbackStore, CSEFeedbackLoader
from training.trainer import TrainingOrchestrator, DatasetBuilder

from scripts.pipeline_utils import (
    fetch_all_pages,
    submit_provider_fetch,
    convert_to_match_record,
    suggestion_to_dict,
    build_match_summary,
    push_results_to_api,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("scheduler")

# Graceful shutdown
_shutdown = Event()


def _handle_signal(signum, frame):
    logger.info(f"Received signal {signum} — shutting down after current cycle...")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


BET365_ENDPOINT = CONFIG.endpoints.bet365_endpoint


# ═══════════════════════════════════════════════
# Phase 1: Self-Training from CSE Feedback
# ═══════════════════════════════════════════════

def phase_self_train(engine: Optional[InferenceEngine] = None) -> dict:
    """Fetch CSE feedback, build training pairs, train models if enough data."""
    logger.info("=" * 60)
    logger.info("PHASE 1: SELF-TRAINING FROM CSE FEEDBACK")
    logger.info("=" * 60)

    sched = CONFIG.scheduler

    if sched.use_local_feedback_api:
        CONFIG.feedback_api.use_local = True

    feedback_rows = CSEFeedbackLoader.fetch_feedback(platform=sched.platform)

    if not feedback_rows:
        logger.info("  No feedback rows from API — skipping training.")
        return {"status": "no_feedback", "feedback_rows": 0}

    logger.info(f"  Fetched {len(feedback_rows)} feedback rows")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    save_json(feedback_rows, os.path.join("data", f"cse_feedback_{timestamp}.json"))

    store = FeedbackStore()
    n_pairs, counts = CSEFeedbackLoader.convert_to_training_pairs(feedback_rows, store)

    trainable = counts["correct"] + counts["not_correct"] + counts["need_to_swap"]
    logger.info(
        f"  Feedback: correct={counts['correct']}, not_correct={counts['not_correct']}, "
        f"swap={counts['need_to_swap']}, skipped={counts['not_sure_skipped']}, "
        f"pairs={n_pairs}"
    )

    min_feedback = CONFIG.feedback_api.min_feedback_for_training
    if trainable < min_feedback:
        logger.info(f"  Insufficient trainable feedback ({trainable} < {min_feedback}) — skipping training.")
        return {"status": "insufficient_data", "feedback_rows": len(feedback_rows), "trainable": trainable}

    all_pairs = store.get_training_pairs()
    if not DatasetBuilder.validate_no_contamination(all_pairs):
        logger.error("  Label contamination detected — aborting training!")
        return {"status": "contamination_error"}

    os.makedirs("models", exist_ok=True)
    sbert_out = f"models/sbert_cse_tuned_{timestamp}"
    ce_out = f"models/ce_cse_tuned_{timestamp}"

    orchestrator = TrainingOrchestrator(all_pairs)
    result = orchestrator.train_all(sbert_out, ce_out, track_accuracy=True)

    if result.get("accuracy_comparison"):
        logger.info(result["accuracy_comparison"])

    # Auto-reload models into the engine (only if training actually produced them)
    sbert_path = result.get("sbert_model_path")
    ce_path = result.get("cross_encoder_model_path")
    models_updated = False

    if sbert_path:
        CONFIG.model.tuned_sbert_path = sbert_path
        CONFIG.model.use_tuned_sbert = True
        models_updated = True
    if ce_path:
        CONFIG.model.tuned_cross_encoder_path = ce_path
        CONFIG.model.use_tuned_cross_encoder = True
        models_updated = True

    if engine and models_updated:
        engine.reload_models()
        logger.info("  Models reloaded into inference engine.")
    elif not models_updated:
        logger.info("  No new models produced — skipping reload.")

    os.makedirs("reports", exist_ok=True)
    save_json(
        {**result, "feedback_counts": counts, "timestamp": timestamp},
        os.path.join("reports", f"self_train_report_{timestamp}.json"),
    )

    logger.info("  Phase 1 complete — models trained and reloaded.")
    return {"status": "trained", "training_result": result, "feedback_counts": counts}


# ═══════════════════════════════════════════════
# Phase 2: Fetch Data from APIs
# ═══════════════════════════════════════════════

def phase_fetch_data() -> Dict[str, Any]:
    """Fetch all enabled provider matches concurrently.

    - Providers with separate_bet365=True (e.g. ODDSPORTAL): fetch Bet365 from a separate API
    - Other providers (SBO, FlashScore, SofaScore): Bet365 is bundled in their API response

    Returns: {
        "PROVIDER_NAME": [...],         # provider matches
        "PROVIDER_NAME_bet365": [...],  # Bet365 matches for that provider
    }
    """
    providers = CONFIG.endpoints.get_active_providers()
    provider_names = [p["name"] for p in providers]

    needs_separate_bet365 = [p for p in providers if p.get("separate_bet365")]

    logger.info("=" * 60)
    logger.info("PHASE 2: FETCHING PROVIDER DATA")
    logger.info(f"  Providers: {', '.join(provider_names)}")
    if needs_separate_bet365:
        names = [p["name"] for p in needs_separate_bet365]
        logger.info(f"  Separate Bet365 fetch needed for: {', '.join(names)}")
    logger.info("=" * 60)

    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=len(providers) + 1) as executor:
        futures = {}

        if needs_separate_bet365:
            futures[executor.submit(
                fetch_all_pages, BET365_ENDPOINT, "Bet365",
                shutdown_event=_shutdown,
            )] = ("_shared_bet365", None)

        for p in providers:
            futures[submit_provider_fetch(
                executor, p, shutdown_event=_shutdown,
            )] = (p["name"], p)

        for future in as_completed(futures):
            label, provider_info = futures[future]
            try:
                raw = future.result()

                if label == "_shared_bet365":
                    shared_b365 = raw["matches"]
                    for p in needs_separate_bet365:
                        results[f"{p['name']}_bet365"] = shared_b365
                    logger.info(f"  [Bet365] fetched {len(shared_b365)} matches (for {', '.join(p['name'] for p in needs_separate_bet365)})")
                    continue

                results[label] = raw["matches"]
                if raw.get("bet365"):
                    results[f"{label}_bet365"] = raw["bet365"]
                    logger.info(
                        f"  [{label}] {len(raw['matches'])} provider matches "
                        f"+ {len(raw['bet365'])} bundled Bet365 matches"
                    )
                else:
                    logger.info(f"  [{label}] {len(raw['matches'])} provider matches")

            except Exception as e:
                logger.error(f"  Failed to fetch {label}: {e}")
                if label != "_shared_bet365":
                    results[label] = []

    data_dir = CONFIG.data_dir
    os.makedirs(data_dir, exist_ok=True)
    for label, rows in results.items():
        if isinstance(rows, list):
            save_json(rows, os.path.join(data_dir, f"{label.lower()}_raw.json"))

    summary_parts = []
    for name in provider_names:
        count = len(results.get(name, []))
        b365_count = len(results.get(f"{name}_bet365", []))
        part = f"{name}={count}"
        if b365_count:
            part += f"(+{b365_count} B365)"
        summary_parts.append(part)
    logger.info(f"  Phase 2 complete — {', '.join(summary_parts)}")

    return results


# ═══════════════════════════════════════════════
# Phase 3: Run Inference + Store Results
# ═══════════════════════════════════════════════

def _log_match_summary(summary: dict, provider_name: str = ""):
    """Log a human-readable match summary."""
    header = f"  MATCH SUMMARY — {provider_name}" if provider_name else "  MATCH SUMMARY"
    logger.info("")
    logger.info("─" * 50)
    logger.info(header)
    logger.info("─" * 50)
    logger.info(f"  Total OP matches processed:  {summary['total_processed']}")
    logger.info("")

    gd = summary["gate_decisions"]
    logger.info(f"  Gate Decisions:")
    logger.info(f"    AUTO_MATCH:   {gd['AUTO_MATCH']}")
    logger.info(f"    NEED_REVIEW:  {gd['NEED_REVIEW']}")
    logger.info("")

    bm = summary["bet365_match"]
    logger.info(f"  B365 Match Status:")
    logger.info(f"    Matched (has bet365_match):    {bm['matched']}")
    logger.info(f"    No match (bet365_match null):  {bm['no_match_null']}")
    logger.info("")

    cd = summary["confidence_distribution"]
    logger.info(f"  Confidence Distribution:")
    logger.info(f"    High   (>= 0.80):  {cd['high_gte_080']}")
    logger.info(f"    Medium (0.50-0.80): {cd['medium_050_080']}")
    logger.info(f"    Low    (< 0.50):    {cd['low_lt_050']}")
    logger.info(f"    No candidate:       {cd['no_candidate']}")
    logger.info("")

    if summary["by_sport"]:
        logger.info(f"  By Sport:")
        for sport, sc in sorted(summary["by_sport"].items(), key=lambda x: x[1]["total"], reverse=True):
            logger.info(
                f"    {sport:<20s}  total={sc['total']:<5d} "
                f"auto={sc['auto_match']:<5d} review={sc['need_review']:<5d} "
                f"no_cand={sc['no_candidate']}"
            )
    logger.info("─" * 50)
    logger.info("")


def phase_inference(engine: InferenceEngine, fetch_data: Dict[str, Any]) -> dict:
    """Run batch inference per provider, using each provider's own Bet365 pool."""
    logger.info("=" * 60)
    logger.info("PHASE 3: RUNNING AI INFERENCE (MULTI-PROVIDER)")
    logger.info("=" * 60)

    current_b365_source = None

    providers = CONFIG.endpoints.get_active_providers()
    threshold = CONFIG.output.confidence_threshold
    provider_reports: Dict[str, Any] = {}
    all_pushable: list = []
    total_processed = 0
    total_null_skipped = 0
    total_elapsed = 0.0

    for provider in providers:
        name = provider["name"]
        raw_data = fetch_data.get(name, [])

        if not raw_data:
            logger.info(f"  [{name}] No data — skipping.")
            provider_reports[name] = {"status": "no_data", "total": 0}
            continue

        provider_b365_raw = fetch_data.get(f"{name}_bet365", [])

        if not provider_b365_raw:
            logger.warning(f"  [{name}] No Bet365 data in API response — skipping.")
            provider_reports[name] = {"status": "no_bet365_data", "total": 0}
            continue

        b365_records = [convert_to_match_record(r, "B365") for r in provider_b365_raw]
        b365_source = f"{name}_bet365"
        logger.info(f"  [{name}] Using Bet365 from {name} API: {len(b365_records)} matches")

        if b365_source != current_b365_source:
            logger.info(f"  Indexing {len(b365_records)} B365 matches (source: {name} API)...")
            engine.index_b365_pool(b365_records)
            current_b365_source = b365_source

        records = [convert_to_match_record(r, name) for r in raw_data]
        logger.info(f"  [{name}] Running inference on {len(records)} matches...")

        start = time.time()
        suggestions = engine.predict_batch(records)
        elapsed = time.time() - start
        rate = len(records) / max(elapsed, 0.001)
        total_elapsed += elapsed

        all_results = [suggestion_to_dict(s, platform=name) for s in suggestions]
        total_processed += len(all_results)

        summary = build_match_summary(suggestions, all_results)
        _log_match_summary(summary, provider_name=name)

        output = [r for r in all_results if r["confidence"] >= threshold and r.get("bet365_match")]
        null_skipped = sum(1 for r in all_results if r["confidence"] >= threshold and not r.get("bet365_match"))
        total_null_skipped += null_skipped
        all_pushable.extend(output)

        logger.info(
            f"  [{name}] {len(all_results)} total, {len(output)} pushable, "
            f"{null_skipped} null-skipped  ({rate:.1f} matches/s, {elapsed:.1f}s)"
        )

        provider_reports[name] = {
            "status": "ok",
            "total_processed": len(all_results),
            "pushable": len(output),
            "null_bet365_skipped": null_skipped,
            "throughput": round(rate, 1),
            "elapsed_seconds": round(elapsed, 1),
            "summary": summary,
        }

    # Combined summary across all providers
    logger.info("")
    logger.info("─" * 50)
    logger.info("  COMBINED RESULTS — ALL PROVIDERS")
    logger.info("─" * 50)
    for name, rpt in provider_reports.items():
        pushable = rpt.get("pushable", 0)
        total = rpt.get("total_processed", 0)
        logger.info(f"    {name:<15s}  processed={total:<6d} pushable={pushable}")
    logger.info(f"    {'TOTAL':<15s}  processed={total_processed:<6d} pushable={len(all_pushable)}")
    logger.info(f"    Null bet365_match skipped: {total_null_skipped}")
    logger.info("─" * 50)

    # Save combined results to local file
    output_path = None
    if CONFIG.output.save_to_file and all_pushable:
        data_dir = CONFIG.data_dir
        output_path = os.path.join(data_dir, "ai_suggested_mappings.json")
        save_json(all_pushable, output_path)
    elif CONFIG.output.save_to_file:
        logger.info("  No pushable results to save.")
    else:
        logger.info("  SAVE_OUTPUT_TO_FILE=false — skipping local file save.")

    # Push combined results to external API
    push_stats = None
    if CONFIG.output.push_to_api and all_pushable:
        push_stats = push_results_to_api(all_pushable)
    elif CONFIG.output.push_to_api:
        logger.info("  No pushable results (all null bet365_match or below threshold).")

    logger.info("  Phase 3 complete.")
    return {
        "status": "ok",
        "providers": provider_reports,
        "total_processed": total_processed,
        "total_pushable": len(all_pushable),
        "total_null_bet365_skipped": total_null_skipped,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "saved_to_file": output_path,
        "push_stats": push_stats,
    }


# ═══════════════════════════════════════════════
# Full Cycle Orchestration
# ═══════════════════════════════════════════════

def run_cycle(engine: InferenceEngine, cycle_num: int) -> dict:
    """Execute one full cycle: train → fetch → infer."""
    cycle_start = time.time()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("")
    logger.info("#" * 60)
    logger.info(f"  CYCLE {cycle_num} STARTED — {timestamp}")
    logger.info("#" * 60)

    report: Dict[str, Any] = {
        "cycle": cycle_num,
        "started_at": timestamp,
        "phases": {},
    }

    sched = CONFIG.scheduler

    # Phase 1: Self-train
    if sched.enable_training:
        try:
            report["phases"]["training"] = phase_self_train(engine)
        except Exception as e:
            logger.error(f"Phase 1 (training) failed: {e}\n{traceback.format_exc()}")
            report["phases"]["training"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Phase 1 (training) — DISABLED via config")
        report["phases"]["training"] = {"status": "disabled"}

    if _shutdown.is_set():
        report["aborted"] = True
        return report

    # Phase 2 + 3: Fetch + Inference
    if sched.enable_inference:
        try:
            data = phase_fetch_data()
        except Exception as e:
            logger.error(f"Phase 2 (fetch) failed: {e}\n{traceback.format_exc()}")
            report["phases"]["fetch"] = {"status": "error", "error": str(e)}
            data = {}

        if _shutdown.is_set():
            report["aborted"] = True
            return report

        has_any_bet365 = any(
            k.endswith("_bet365") and data.get(k) for k in data
        )
        if has_any_bet365:
            try:
                report["phases"]["inference"] = phase_inference(engine, data)
            except Exception as e:
                logger.error(f"Phase 3 (inference) failed: {e}\n{traceback.format_exc()}")
                report["phases"]["inference"] = {"status": "error", "error": str(e)}
        else:
            report["phases"]["inference"] = {"status": "no_bet365_data"}
    else:
        logger.info("Phase 2+3 (inference) — DISABLED via config")
        report["phases"]["fetch"] = {"status": "disabled"}
        report["phases"]["inference"] = {"status": "disabled"}

    elapsed = time.time() - cycle_start
    report["elapsed_seconds"] = round(elapsed, 1)
    report["finished_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Save cycle report
    os.makedirs(CONFIG.logs_dir, exist_ok=True)
    report_path = os.path.join(CONFIG.logs_dir, f"cycle_{cycle_num}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    save_json(report, report_path)

    logger.info("")
    logger.info(f"  CYCLE {cycle_num} FINISHED in {elapsed:.1f}s")
    logger.info(f"  Report: {report_path}")
    logger.info("#" * 60)

    return report


# ═══════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Automated pipeline scheduler")
    parser.add_argument("--run-once", action="store_true",
                        help="Run a single cycle and exit (no loop)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Disable Phase 1 (self-training) for this run")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Disable Phase 2+3 (fetch + inference) for this run")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override interval in minutes (default: from env/config)")
    args = parser.parse_args()

    sched = CONFIG.scheduler
    interval = args.interval or sched.interval_minutes

    if args.skip_training:
        sched.enable_training = False
    if args.skip_inference:
        sched.enable_inference = False

    providers = CONFIG.endpoints.get_active_providers()
    provider_names = [p["name"] for p in providers]

    import torch

    print("\n" + "=" * 60)
    print("  AI MATCH MAPPING ENGINE — AUTOMATED SCHEDULER")
    print("=" * 60)
    out = CONFIG.output

    device = CONFIG.model.device
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU:              {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  CUDA Version:     {torch.version.cuda}")
        print(f"  FP16 Training:    ON")
        print(f"  FP16 Inference:   ON")
    else:
        print(f"  Device:           {device} (no GPU acceleration)")

    print(f"  Interval:         {interval} minutes")
    print(f"  Providers:        {', '.join(provider_names)}")
    print(f"  Training:         {'ON' if sched.enable_training else 'OFF'}")
    print(f"    SBERT batch:    {CONFIG.training.sbert_batch_size}")
    print(f"    CE batch:       {CONFIG.training.ce_batch_size} x{CONFIG.training.gradient_accumulation_steps} grad_accum")
    print(f"  Inference:        {'ON' if sched.enable_inference else 'OFF'}")
    print(f"    Encode batch:   {CONFIG.model.encode_batch_size}")
    print(f"    Rerank batch:   {CONFIG.model.rerank_batch_size}")
    print(f"  Save to file:     {'ON' if out.save_to_file else 'OFF'}")
    print(f"  Push to API:      {'ON' if out.push_to_api else 'OFF'}")
    if out.push_to_api:
        print(f"  Push URL:         {CONFIG.endpoints.store_results_url}")
    print(f"  Confidence:       >= {out.confidence_threshold}")
    print(f"  Press Ctrl+C to stop gracefully.")
    print("=" * 60 + "\n")

    logger.info("Initializing inference engine...")
    engine = InferenceEngine()
    logger.info("Engine ready.")

    cycle_num = 0

    if args.run_once:
        cycle_num += 1
        run_cycle(engine, cycle_num)
        print("\n  Single cycle complete. Exiting.")
        return

    while not _shutdown.is_set():
        cycle_num += 1
        run_cycle(engine, cycle_num)

        if _shutdown.is_set():
            break

        next_run = datetime.utcnow().strftime("%H:%M:%S")
        logger.info(f"  Next cycle in {interval} minutes (sleeping until then)...")
        logger.info(f"  Current time: {next_run}")

        # Sleep in small increments so we can respond to shutdown signals quickly
        sleep_seconds = interval * 60
        slept = 0
        while slept < sleep_seconds and not _shutdown.is_set():
            chunk = min(5, sleep_seconds - slept)
            _shutdown.wait(timeout=chunk)
            slept += chunk

    logger.info("Scheduler shut down cleanly.")


if __name__ == "__main__":
    main()

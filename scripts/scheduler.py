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
import json
import time
import signal
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Event
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.models import MatchRecord
from core.inference import InferenceEngine
from core.normalizer import detect_categories
from core.feedback import FeedbackStore, CSEFeedbackLoader
from training.trainer import TrainingOrchestrator, DatasetBuilder
from evaluation.accuracy_tracker import AccuracyTracker, ComparisonReport

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


# ═══════════════════════════════════════════════
# API Helpers (shared with api_pipeline)
# ═══════════════════════════════════════════════

BET365_ENDPOINT = CONFIG.endpoints.bet365_endpoint
PAGE_LIMIT = 100

SPORT_ID_MAP = {
    "1": "soccer", "2": "tennis", "3": "basketball", "4": "hockey",
    "5": "volleyball", "6": "handball", "7": "baseball",
    "8": "american_football", "9": "rugby", "10": "boxing",
    "12": "table_tennis", "13": "cricket", "18": "esports",
}


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_all_pages(
    endpoint: str, label: str, data_key: str = "rows", bet365_key: str = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all pages from a paginated API. Also extracts bundled Bet365 from first page.

    Returns dict: {"matches": [...], "bet365": [...]}
    """
    session = _build_session()
    all_rows: list = []
    b365_rows: list = []
    page = 1
    consecutive_failures = 0

    while not _shutdown.is_set():
        try:
            resp = session.get(endpoint, params={"page": page, "limit": PAGE_LIMIT}, timeout=60)
            resp.raise_for_status()
            consecutive_failures = 0
        except requests.RequestException as e:
            consecutive_failures += 1
            logger.warning(f"[{label}] page {page} failed (attempt {consecutive_failures}): {e}")
            if consecutive_failures >= 3:
                break
            time.sleep(min(2 ** consecutive_failures, 30))
            continue

        body = resp.json()
        if not body.get("status"):
            logger.error(f"[{label}] API error: {body.get('message')}")
            break

        data = body["data"]
        rows = data.get(data_key, [])
        total_pages = data.get("totalPages", 1)
        all_rows.extend(rows)

        if page == 1 and bet365_key and isinstance(data, dict):
            b365_rows = data.get(bet365_key, [])
            if b365_rows:
                logger.info(f"  [{label}] found {len(b365_rows)} bundled Bet365 matches in response")

        logger.info(f"  [{label}] page {page}/{total_pages} — {len(all_rows)} rows so far")

        if page >= total_pages or not rows:
            break
        page += 1
        time.sleep(0.3)

    logger.info(f"  [{label}] fetched {len(all_rows)} matches total")
    return {"matches": all_rows, "bet365": b365_rows}


def fetch_single_response(
    endpoint: str, label: str, data_key: str, bet365_key: str = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch from a non-paginated API that returns all data in one response.

    Expected format: { "status": true, "data": { "<data_key>": [...], "bet365Matches": [...] } }

    Returns dict: {"matches": [...], "bet365": [...]}
    """
    session = _build_session()
    try:
        resp = session.get(endpoint, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"  [{label}] API request failed: {e}")
        return {"matches": [], "bet365": []}

    body = resp.json()
    if not body.get("status"):
        logger.error(f"  [{label}] API error: {body.get('message')}")
        return {"matches": [], "bet365": []}

    data = body.get("data", {})
    if isinstance(data, list):
        rows = data
        b365_rows = []
    elif isinstance(data, dict):
        rows = data.get(data_key, [])
        b365_rows = data.get(bet365_key, []) if bet365_key else []
        if not rows:
            available_keys = [k for k, v in data.items() if isinstance(v, list)]
            logger.warning(
                f"  [{label}] Key '{data_key}' returned 0 rows. "
                f"Available list keys in response: {available_keys}"
            )
    else:
        rows = []
        b365_rows = []

    logger.info(
        f"  [{label}] fetched {len(rows)} provider matches"
        + (f" + {len(b365_rows)} bet365 matches" if b365_rows else "")
    )
    return {"matches": rows, "bet365": b365_rows}


def convert_to_match_record(raw: Dict[str, Any], platform_tag: str) -> MatchRecord:
    sport_id = str(raw.get("sport_id", ""))
    sport_name = raw.get("sport", SPORT_ID_MAP.get(sport_id, "unknown")).lower()

    league_obj = raw.get("league") or {}
    league_name = league_obj.get("name", "") if isinstance(league_obj, dict) else str(league_obj)

    home = raw.get("home_team", "")
    away = raw.get("away_team", "")

    commence = raw.get("commence_time")
    if isinstance(commence, (int, float)):
        kickoff = datetime.fromtimestamp(commence, tz=timezone.utc)
    else:
        kickoff = datetime.now(tz=timezone.utc)

    full_text = f"{league_name} {home} {away}"
    cats = detect_categories(full_text)

    return MatchRecord(
        match_id=str(raw.get("id", "")),
        platform=platform_tag,
        sport=sport_name,
        league=league_name,
        home_team=home,
        away_team=away,
        kickoff=kickoff,
        category_tags=cats,
    )


CONFIDENCE_THRESHOLD = CONFIG.output.confidence_threshold


def suggestion_to_dict(suggestion, platform: str = "ODDSPORTAL") -> Dict[str, Any]:
    top = suggestion.candidates_top5[0] if suggestion.candidates_top5 else None
    score = top.score if top else 0.0
    gate = suggestion.gate_decision.value if hasattr(suggestion.gate_decision, "value") else str(suggestion.gate_decision)

    reason = "no_candidates"
    if gate == "AUTO_MATCH":
        reason = "rule_based_strong_match"
    elif score >= 0.80:
        reason = "ai_high_confidence"
    elif score >= CONFIDENCE_THRESHOLD:
        reason = "ai_moderate_confidence"
    elif score > 0:
        reason = "ai_low_confidence"

    return {
        "platform": platform,
        "bet365_match": top.b365_match_id if top else None,
        "provider_id": suggestion.op_match_id,
        "confidence": round(score, 2),
        "is_checked": False,
        "is_mapped": score >= CONFIDENCE_THRESHOLD and top is not None,
        "reason": reason,
        "switch": top.swapped if top else False,
    }


def save_json(data: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Saved → {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")


def push_results_to_api(results: List[dict]) -> dict:
    """POST each AI-suggested mapping in parallel using a thread pool."""
    url = CONFIG.endpoints.store_results_url
    if not url:
        return {"pushed": 0, "failed": 0, "total": len(results)}

    total = len(results)
    workers = CONFIG.output.push_workers
    session = _build_session()

    pushed = 0
    failed = 0
    lock = __import__("threading").Lock()
    log_interval = max(1, total // 20)

    logger.info(f"  Pushing {total} results to {url}  (workers={workers})...")

    def _push_one(item):
        idx, result = item
        try:
            resp = session.post(url, json=result, timeout=30)
            resp.raise_for_status()
            return idx, True, None
        except requests.RequestException as e:
            return idx, False, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, success, err in pool.map(_push_one, enumerate(results)):
            with lock:
                if success:
                    pushed += 1
                else:
                    failed += 1
                    logger.warning(
                        f"  FAILED [{idx + 1}/{total}] provider_id={results[idx].get('provider_id')}: {err}"
                    )

                done = pushed + failed
                pending = total - done
                if done % log_interval == 0 or done == total:
                    logger.info(
                        f"  Progress: {done}/{total}  "
                        f"(pushed={pushed}, failed={failed}, pending={pending})"
                    )

    logger.info(
        f"  Push complete: {pushed} pushed, {failed} failed, {total} total"
    )
    return {"pushed": pushed, "failed": failed, "total": total}


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

def _submit_provider_fetch(executor, provider_info: dict):
    """Submit the right fetch function based on provider config."""
    name = provider_info["name"]
    endpoint = provider_info["endpoint"]
    data_key = provider_info.get("data_key", "rows")
    bet365_key = provider_info.get("bet365_key")
    if provider_info.get("paginated", True):
        return executor.submit(fetch_all_pages, endpoint, name, data_key, bet365_key)
    else:
        return executor.submit(fetch_single_response, endpoint, name, data_key, bet365_key)


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
                fetch_all_pages, BET365_ENDPOINT, "Bet365"
            )] = ("_shared_bet365", None)

        for p in providers:
            futures[_submit_provider_fetch(executor, p)] = (p["name"], p)

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

def _build_match_summary(suggestions, all_results: list) -> dict:
    """Build a structured summary from suggestions and flat result dicts."""
    total = len(all_results)

    auto_match = sum(1 for s in suggestions if s.gate_decision.value == "AUTO_MATCH")
    need_review = total - auto_match

    has_b365 = sum(1 for r in all_results if r.get("bet365_match"))
    no_b365 = total - has_b365

    high = sum(1 for r in all_results if r["confidence"] >= 0.80 and r.get("bet365_match"))
    medium = sum(1 for r in all_results if 0.50 <= r["confidence"] < 0.80 and r.get("bet365_match"))
    low = sum(1 for r in all_results if 0 < r["confidence"] < 0.50 and r.get("bet365_match"))

    sport_counts: dict = {}
    for s in suggestions:
        sport = s.op_sport.lower()
        sport_counts.setdefault(sport, {"total": 0, "auto_match": 0, "need_review": 0, "no_candidate": 0})
        sport_counts[sport]["total"] += 1
        if not s.candidates_top5:
            sport_counts[sport]["no_candidate"] += 1
        elif s.gate_decision.value == "AUTO_MATCH":
            sport_counts[sport]["auto_match"] += 1
        else:
            sport_counts[sport]["need_review"] += 1

    return {
        "total_processed": total,
        "gate_decisions": {"AUTO_MATCH": auto_match, "NEED_REVIEW": need_review},
        "bet365_match": {"matched": has_b365, "no_match_null": no_b365},
        "confidence_distribution": {
            "high_gte_080": high,
            "medium_050_080": medium,
            "low_lt_050": low,
            "no_candidate": no_b365,
        },
        "by_sport": sport_counts,
    }


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

        summary = _build_match_summary(suggestions, all_results)
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

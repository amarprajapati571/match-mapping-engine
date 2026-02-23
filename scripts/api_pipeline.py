"""
Live API Integration Pipeline.

Fetches real match data from the sports-bet API, runs AI mapping inference,
and outputs suggested mappings to a JSON file for review.

Performance:
  - Concurrent Bet365 + OddsPortal fetching via ThreadPoolExecutor
  - Batch-optimized inference (single SBERT encode + single CE scoring call)

Run: python scripts/api_pipeline.py
"""

import sys
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.models import MatchRecord
from core.inference import InferenceEngine
from core.normalizer import detect_categories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("api_pipeline")

BET365_ENDPOINT = CONFIG.endpoints.bet365_endpoint
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PAGE_LIMIT = 100


# ═══════════════════════════════════════════════
# API Fetching
# ═══════════════════════════════════════════════

def _build_session() -> requests.Session:
    """Build a requests session with automatic retries on transient errors."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
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
    all_rows = []
    b365_rows = []
    page = 1
    consecutive_failures = 0
    max_consecutive_failures = 3

    logger.info(f"Fetching {label} matches from API...")

    while True:
        params = {"page": page, "limit": PAGE_LIMIT}
        try:
            resp = session.get(endpoint, params=params, timeout=60)
            resp.raise_for_status()
            consecutive_failures = 0
        except requests.RequestException as e:
            consecutive_failures += 1
            logger.warning(f"Request failed on page {page} (attempt {consecutive_failures}): {e}")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Stopping after {max_consecutive_failures} consecutive failures")
                break
            wait = min(2 ** consecutive_failures, 30)
            logger.info(f"  Waiting {wait}s before retry...")
            time.sleep(wait)
            continue

        body = resp.json()
        if not body.get("status"):
            logger.error(f"API returned error: {body.get('message')}")
            break

        data = body["data"]
        rows = data.get(data_key, [])
        total_count = data.get("totalCount", 0)
        total_pages = data.get("totalPages", 1)

        all_rows.extend(rows)

        if page == 1 and bet365_key and isinstance(data, dict):
            b365_rows = data.get(bet365_key, [])
            if b365_rows:
                logger.info(f"  [{label}] found {len(b365_rows)} bundled Bet365 matches in response")

        logger.info(
            f"  [{label}] Page {page}/{total_pages} — "
            f"fetched {len(rows)} rows (total so far: {len(all_rows)}/{total_count})"
        )

        if page >= total_pages or not rows:
            break

        page += 1
        time.sleep(0.3)

    logger.info(f"  [{label}] Done — {len(all_rows)} matches total")
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


def save_json(data: Any, filepath: str):
    """Write data to a JSON file with pretty formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Saved → {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")


# ═══════════════════════════════════════════════
# Data Conversion
# ═══════════════════════════════════════════════

SPORT_ID_MAP = {
    "1": "soccer",
    "2": "tennis",
    "3": "basketball",
    "4": "hockey",
    "5": "volleyball",
    "6": "handball",
    "7": "baseball",
    "8": "american_football",
    "9": "rugby",
    "10": "boxing",
    "12": "table_tennis",
    "13": "cricket",
    "18": "esports",
}


def convert_to_match_record(raw: Dict[str, Any], platform_tag: str) -> MatchRecord:
    """Convert a raw API match dict into the engine's MatchRecord format."""
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


# ═══════════════════════════════════════════════
# Serialization helpers
# ═══════════════════════════════════════════════

CONFIDENCE_THRESHOLD = CONFIG.output.confidence_threshold


def _push_results_to_api(results: list) -> dict:
    """POST each result in parallel using a thread pool with live progress."""
    url = CONFIG.endpoints.store_results_url
    if not url:
        return {"pushed": 0, "failed": 0, "total": len(results)}

    import threading

    total = len(results)
    workers = CONFIG.output.push_workers
    session = _build_session()

    pushed = 0
    failed = 0
    lock = threading.Lock()
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


def _gate_to_reason(gate_decision: str, score: float) -> str:
    if gate_decision == "AUTO_MATCH":
        return "rule_based_strong_match"
    if score >= 0.80:
        return "ai_high_confidence"
    if score >= CONFIDENCE_THRESHOLD:
        return "ai_moderate_confidence"
    if score > 0:
        return "ai_low_confidence"
    return "no_candidates"


def suggestion_to_dict(suggestion, platform: str = "ODDSPORTAL") -> Dict[str, Any]:
    """Convert a MappingSuggestion to the flat output format."""
    top = suggestion.candidates_top5[0] if suggestion.candidates_top5 else None
    score = top.score if top else 0.0

    return {
        "platform": platform,
        "bet365_match": top.b365_match_id if top else None,
        "provider_id": suggestion.op_match_id,
        "confidence": round(score, 2),
        "is_checked": False,
        "is_mapped": score >= CONFIDENCE_THRESHOLD and top is not None,
        "reason": _gate_to_reason(suggestion.gate_decision.value, score),
        "switch": top.swapped if top else False,
    }


# ═══════════════════════════════════════════════
# Main Pipeline
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


def _print_match_summary(summary: dict, provider_name: str = ""):
    """Print a human-readable match summary."""
    header = f"  MATCH SUMMARY — {provider_name}" if provider_name else "  MATCH SUMMARY"
    print("")
    print("─" * 50)
    print(header)
    print("─" * 50)
    print(f"  Total OP matches processed:  {summary['total_processed']}")
    print("")

    gd = summary["gate_decisions"]
    print(f"  Gate Decisions:")
    print(f"    AUTO_MATCH:   {gd['AUTO_MATCH']}")
    print(f"    NEED_REVIEW:  {gd['NEED_REVIEW']}")
    print("")

    bm = summary["bet365_match"]
    print(f"  B365 Match Status:")
    print(f"    Matched (has bet365_match):    {bm['matched']}")
    print(f"    No match (bet365_match null):  {bm['no_match_null']}")
    print("")

    cd = summary["confidence_distribution"]
    print(f"  Confidence Distribution:")
    print(f"    High   (>= 0.80):  {cd['high_gte_080']}")
    print(f"    Medium (0.50-0.80): {cd['medium_050_080']}")
    print(f"    Low    (< 0.50):    {cd['low_lt_050']}")
    print(f"    No candidate:       {cd['no_candidate']}")
    print("")

    if summary["by_sport"]:
        print(f"  By Sport:")
        for sport, sc in sorted(summary["by_sport"].items(), key=lambda x: x[1]["total"], reverse=True):
            print(
                f"    {sport:<20s}  total={sc['total']:<5d} "
                f"auto={sc['auto_match']:<5d} review={sc['need_review']:<5d} "
                f"no_cand={sc['no_candidate']}"
            )
    print("─" * 50)
    print("")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live API → AI Mapping Pipeline (Multi-Provider)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Reuse previously fetched data from data/*.json instead of calling APIs")
    args = parser.parse_args()

    providers = CONFIG.endpoints.get_active_providers()
    provider_names = [p["name"] for p in providers]

    print("\n" + "=" * 70)
    print("  LIVE API → AI MAPPING PIPELINE (MULTI-PROVIDER)")
    print("=" * 70)
    print(f"  Providers: {', '.join(provider_names)}")
    print("=" * 70 + "\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1-2: Fetch all providers ──
    # ODDSPORTAL: Bet365 from separate API  |  Others: Bet365 bundled in same API
    fetch_data: dict = {}
    needs_separate_bet365 = [p for p in providers if p.get("separate_bet365")]

    if args.skip_fetch:
        print("Step 1-2: Loading cached data (--skip-fetch)...")
        for p in providers:
            name = p["name"]
            path = os.path.join(DATA_DIR, f"{name.lower()}_raw.json")
            try:
                with open(path, "r") as f:
                    fetch_data[name] = json.load(f)
            except FileNotFoundError:
                fetch_data[name] = []
            b365_path_prov = os.path.join(DATA_DIR, f"{name.lower()}_bet365_raw.json")
            try:
                with open(b365_path_prov, "r") as f:
                    b365_data = json.load(f)
                    if b365_data:
                        fetch_data[f"{name}_bet365"] = b365_data
            except FileNotFoundError:
                pass
            b365_count = len(fetch_data.get(f'{name}_bet365', []))
            b365_info = f" (+{b365_count} B365)" if b365_count else ""
            print(f"  {name}: {len(fetch_data[name])} matches{b365_info}")
    else:
        print(f"Step 1-2: Fetching {len(providers)} providers concurrently...")
        if needs_separate_bet365:
            sep_names = [p["name"] for p in needs_separate_bet365]
            print(f"  Separate Bet365 fetch for: {', '.join(sep_names)}")

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
                            fetch_data[f"{p['name']}_bet365"] = shared_b365
                        print(f"  [Bet365] fetched {len(shared_b365)} matches (for {', '.join(p['name'] for p in needs_separate_bet365)})")
                        continue

                    fetch_data[label] = raw["matches"]
                    if raw.get("bet365"):
                        fetch_data[f"{label}_bet365"] = raw["bet365"]
                        print(
                            f"  [{label}] {len(raw['matches'])} provider matches "
                            f"+ {len(raw['bet365'])} bundled Bet365 matches"
                        )
                    else:
                        print(f"  [{label}] {len(raw['matches'])} provider matches")

                except Exception as e:
                    logger.error(f"Failed to fetch {label}: {e}")
                    if label != "_shared_bet365":
                        fetch_data[label] = []

        for label, rows in fetch_data.items():
            if isinstance(rows, list):
                save_json(rows, os.path.join(DATA_DIR, f"{label.lower()}_raw.json"))

    # ── Step 3-4: Init engine and run per-provider inference ──
    print(f"\nStep 3: Initializing AI inference engine...")
    engine = InferenceEngine()

    current_b365_source = None

    # ── Step 4: Run inference per provider (each uses its own Bet365 pool) ──
    all_pushable: list = []
    provider_stats: dict = {}
    total_processed = 0
    total_null_skipped = 0

    for p in providers:
        name = p["name"]
        raw = fetch_data.get(name, [])
        if not raw:
            print(f"\n  [{name}] No data — skipping.")
            provider_stats[name] = {"total": 0, "pushable": 0}
            continue

        provider_b365_raw = fetch_data.get(f"{name}_bet365", [])

        if not provider_b365_raw:
            print(f"\n  [{name}] No Bet365 data in API response — skipping.")
            provider_stats[name] = {"total": 0, "pushable": 0}
            continue

        b365_records = [convert_to_match_record(r, "B365") for r in provider_b365_raw]
        b365_source = f"{name}_bet365"
        print(f"\n  [{name}] Using Bet365 from {name} API: {len(b365_records)} matches")

        if b365_source != current_b365_source:
            print(f"  Indexing {len(b365_records)} B365 matches (source: {name} API)...")
            engine.index_b365_pool(b365_records)
            current_b365_source = b365_source

        records = [convert_to_match_record(r, name) for r in raw]
        print(f"\n  Step 4 [{name}]: Running inference on {len(records)} matches...")

        start = time.time()
        suggestions = engine.predict_batch(records)
        elapsed = time.time() - start
        rate = len(records) / max(elapsed, 0.001)
        print(f"  Inference complete in {elapsed:.1f}s ({rate:.1f} matches/s)")

        all_results = [suggestion_to_dict(s, platform=name) for s in suggestions]
        total_processed += len(all_results)

        summary = _build_match_summary(suggestions, all_results)
        _print_match_summary(summary, provider_name=name)

        output = [r for r in all_results if r["confidence"] >= CONFIDENCE_THRESHOLD and r.get("bet365_match")]
        null_skipped = sum(1 for r in all_results if r["confidence"] >= CONFIDENCE_THRESHOLD and not r.get("bet365_match"))
        total_null_skipped += null_skipped
        all_pushable.extend(output)

        provider_stats[name] = {
            "total": len(all_results),
            "pushable": len(output),
            "null_skipped": null_skipped,
            "rate": round(rate, 1),
            "summary": summary,
        }

    # ── Step 6: Save and push combined results ──
    print(f"\nStep 6: Filtering combined results...")
    print(f"  Total processed (all providers): {total_processed}")
    print(f"  Total pushable:                  {len(all_pushable)}")
    print(f"  Total null bet365_match skipped: {total_null_skipped}")

    output_path = None
    if CONFIG.output.save_to_file and all_pushable:
        print(f"\nStep 6a: Saving {len(all_pushable)} results to file...")
        output_path = os.path.join(DATA_DIR, "ai_suggested_mappings.json")
        save_json(all_pushable, output_path)
    elif CONFIG.output.save_to_file:
        print("\nStep 6a: No pushable results to save.")
    else:
        print("\nStep 6a: SAVE_OUTPUT_TO_FILE=false — skipping local file save.")

    push_stats = None
    if CONFIG.output.push_to_api and all_pushable:
        print(f"\nStep 6b: Pushing {len(all_pushable)} results to API (parallel)...")
        push_stats = _push_results_to_api(all_pushable)
    elif CONFIG.output.push_to_api:
        print("\nStep 6b: No pushable results (all null bet365_match or below threshold).")
    else:
        print("\nStep 6b: PUSH_RESULTS_TO_API=false — skipping API push.")

    # ── Step 7: Final report ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — MULTI-PROVIDER")
    print("=" * 70)
    print(f"\n  Data fetched:")
    for name in provider_names:
        b365_count = len(fetch_data.get(f"{name}_bet365", []))
        b365_info = f" (+{b365_count} Bet365)" if b365_count else " (no Bet365)"
        print(f"    {name}: {len(fetch_data.get(name, []))} matches{b365_info}")

    print(f"\n  Per-Provider Results:")
    for name, stats in provider_stats.items():
        print(f"    {name:<15s}  total={stats['total']:<6d} pushable={stats['pushable']:<6d} null_skipped={stats.get('null_skipped', 0)}")
    print(f"    {'TOTAL':<15s}  total={total_processed:<6d} pushable={len(all_pushable)}")

    print(f"\n  Output:")
    print(f"    Save to file: {'ON' if CONFIG.output.save_to_file else 'OFF'}{f' -> {output_path}' if output_path else ''}")
    print(f"    Push to API:  {'ON' if CONFIG.output.push_to_api else 'OFF'}{f' -> {push_stats}' if push_stats else ''}")

    print(f"\n  Sample output (first 5):")
    print("  " + "-" * 66)
    for r in all_pushable[:5]:
        print(f"  {json.dumps(r)}")
    print()


if __name__ == "__main__":
    main()

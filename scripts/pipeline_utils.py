"""
Shared utilities for pipeline scripts (scheduler, api_pipeline).

Centralizes:
- HTTP session management with connection pooling and retries
- Paginated and single-response API fetching
- MatchRecord conversion from raw API dicts
- Suggestion serialization and result pushing
- Match summary building
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import CONFIG
from core.models import MatchRecord
from core.normalizer import detect_categories

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════

SPORT_ID_MAP = {
    "1": "soccer", "2": "tennis", "3": "basketball", "4": "hockey",
    "5": "volleyball", "6": "handball", "7": "baseball",
    "8": "american_football", "9": "rugby", "10": "boxing",
    "12": "table_tennis", "13": "cricket", "18": "esports",
}

PAGE_LIMIT = 100

CONFIDENCE_THRESHOLD = CONFIG.output.confidence_threshold


# ═══════════════════════════════════════════════
# HTTP Session (reusable, thread-safe)
# ═══════════════════════════════════════════════

_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Return a reusable thread-safe session with retry logic.

    Sessions are designed to be reused across multiple requests —
    this avoids TCP handshake overhead and enables HTTP keep-alive.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=20,
            pool_maxsize=20,
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def reset_session():
    """Reset the shared session (e.g. after a long idle period)."""
    global _session
    if _session is not None:
        _session.close()
        _session = None


# ═══════════════════════════════════════════════
# API Fetching
# ═══════════════════════════════════════════════

def fetch_all_pages(
    endpoint: str,
    label: str,
    data_key: str = "rows",
    bet365_key: str = None,
    shutdown_event=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all pages from a paginated API.

    Also extracts bundled Bet365 from the first page if bet365_key is provided.
    Returns dict: {"matches": [...], "bet365": [...]}
    """
    session = get_session()
    all_rows: list = []
    b365_rows: list = []
    page = 1
    consecutive_failures = 0

    while True:
        if shutdown_event and shutdown_event.is_set():
            break

        try:
            resp = session.get(
                endpoint, params={"page": page, "limit": PAGE_LIMIT}, timeout=60,
            )
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
    endpoint: str,
    label: str,
    data_key: str,
    bet365_key: str = None,
    shutdown_event=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch from a non-paginated API returning all data in one response.

    Returns dict: {"matches": [...], "bet365": [...]}
    """
    session = get_session()
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


def submit_provider_fetch(executor, provider_info: dict, shutdown_event=None):
    """Submit the right fetch function based on provider config."""
    name = provider_info["name"]
    endpoint = provider_info["endpoint"]
    data_key = provider_info.get("data_key", "rows")
    bet365_key = provider_info.get("bet365_key")
    if provider_info.get("paginated", True):
        return executor.submit(
            fetch_all_pages, endpoint, name, data_key, bet365_key, shutdown_event,
        )
    else:
        return executor.submit(
            fetch_single_response, endpoint, name, data_key, bet365_key, shutdown_event,
        )


# ═══════════════════════════════════════════════
# Data Conversion
# ═══════════════════════════════════════════════

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
# Result Serialization
# ═══════════════════════════════════════════════

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
    gate = (
        suggestion.gate_decision.value
        if hasattr(suggestion.gate_decision, "value")
        else str(suggestion.gate_decision)
    )

    return {
        "platform": platform,
        "bet365_match": top.b365_match_id if top else None,
        "provider_id": suggestion.op_match_id,
        "confidence": round(score, 2),
        "is_checked": False,
        "is_mapped": score >= CONFIDENCE_THRESHOLD and top is not None,
        "reason": _gate_to_reason(gate, score),
        "switch": top.swapped if top else False,
    }


# ═══════════════════════════════════════════════
# Result Pushing
# ═══════════════════════════════════════════════

def push_results_to_api(results: List[dict]) -> dict:
    """POST each AI-suggested mapping in parallel using a thread pool."""
    import threading

    url = CONFIG.endpoints.store_results_url
    if not url:
        return {"pushed": 0, "failed": 0, "total": len(results)}

    total = len(results)
    workers = CONFIG.output.push_workers
    session = get_session()

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
                        f"  FAILED [{idx + 1}/{total}] "
                        f"provider_id={results[idx].get('provider_id')}: {err}"
                    )

                done = pushed + failed
                if done % log_interval == 0 or done == total:
                    pending = total - done
                    logger.info(
                        f"  Progress: {done}/{total}  "
                        f"(pushed={pushed}, failed={failed}, pending={pending})"
                    )

    logger.info(f"  Push complete: {pushed} pushed, {failed} failed, {total} total")
    return {"pushed": pushed, "failed": failed, "total": total}


# ═══════════════════════════════════════════════
# Match Summary
# ═══════════════════════════════════════════════

def build_match_summary(suggestions, all_results: list) -> dict:
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
        sport_counts.setdefault(
            sport, {"total": 0, "auto_match": 0, "need_review": 0, "no_candidate": 0},
        )
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


# ═══════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════

def save_json(data: Any, filepath: str):
    """Write data to a JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Saved → {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")

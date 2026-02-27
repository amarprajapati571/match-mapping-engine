"""
Dynamic League Alias Loader — loads league data from JSON file or external API
and builds alias map for league name resolution.

Supports two data sources (priority order):
  1. Local JSON file  → set LEAGUES_JSON_FILE=data/leagues.json
  2. External API     → set LEAGUES_API_URL=https://...

Each league document has:
  - league_name_en: "J2 League"
  - country_en: "Japan"
  - league_abbreviation: "J2"
  - keywords: ["JAPAN J2 LEAGUE", "Japan J2-League"]

All variants (abbreviation, keywords) are mapped to a canonical form:
  "{country} {league_name}" → e.g., "japan j2 league"

This allows "J2" vs "Japan J2 League" to resolve to similarity = 1.0.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests

from config.settings import CONFIG
from core.normalizer import LEAGUE_ALIASES, clean_text, clear_caches

logger = logging.getLogger(__name__)


def _parse_json_data(data) -> list:
    """Extract league list from various JSON response formats."""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return (
            data.get("leagues")
            or data.get("data")
            or data.get("rows")
            or data.get("results")
            or []
        )
    return []


def load_leagues_from_file(file_path: str = None) -> list:
    """
    Load league data from a local JSON file.

    The JSON file can be:
      - A plain array:  [{"league_name_en": "...", ...}, ...]
      - A wrapped dict:  {"leagues": [...]} or {"data": [...]}

    Returns raw list of league documents.
    """
    path = file_path or os.getenv("LEAGUES_JSON_FILE", "")
    if not path:
        return []

    resolved = Path(path)
    if not resolved.is_absolute():
        # Resolve relative to project root
        resolved = Path(__file__).resolve().parent.parent / path

    if not resolved.exists():
        logger.warning(f"Leagues JSON file not found: {resolved}")
        return []

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        leagues = _parse_json_data(data)
        logger.info(f"Loaded {len(leagues)} leagues from file: {resolved}")
        return leagues
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read leagues JSON file: {e}")
        return []


def fetch_leagues_from_api(url: str = None, timeout: int = 30) -> list:
    """
    Fetch all leagues from the external API.
    Supports optional auth via LEAGUES_API_TOKEN env var.
    Returns raw list of league documents.
    """
    api_url = url or CONFIG.endpoints.leagues_api_url
    logger.info(f"Fetching leagues from API: {api_url}")

    headers = {}
    token = os.getenv("LEAGUES_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        leagues = _parse_json_data(data)
        logger.info(f"Fetched {len(leagues)} leagues from API")
        return leagues

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch leagues from API: {e}")
        return []


def build_alias_map(leagues: list) -> Dict[str, str]:
    """
    Build a league alias map from raw league documents.

    For each league, creates entries:
      abbreviation → canonical
      each keyword  → canonical

    Canonical form: "{country} {league_name}" (lowercased, cleaned)
    """
    alias_map = {}
    skipped = 0

    for league in leagues:
        league_name = league.get("league_name_en", "").strip()
        country = league.get("country_en", "").strip()
        abbreviation = league.get("league_abbreviation", "").strip()
        keywords = league.get("keywords", [])

        if not league_name:
            skipped += 1
            continue

        # Canonical: "country league_name" (e.g., "japan j2 league")
        if country:
            canonical = clean_text(f"{country} {league_name}")
        else:
            canonical = clean_text(league_name)

        # Map abbreviation → canonical
        if abbreviation:
            abbr_clean = clean_text(abbreviation)
            if abbr_clean and abbr_clean != canonical:
                alias_map[abbr_clean] = canonical

        # Map each keyword → canonical
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                kw_clean = clean_text(kw)
                if kw_clean and kw_clean != canonical:
                    alias_map[kw_clean] = canonical

        # Also map the league_name alone → canonical (in case no country prefix)
        name_clean = clean_text(league_name)
        if name_clean and name_clean != canonical:
            alias_map[name_clean] = canonical

    logger.info(
        f"Built {len(alias_map)} league aliases from {len(leagues)} leagues "
        f"({skipped} skipped)"
    )
    return alias_map


def load_league_aliases(url: str = None, file_path: str = None) -> int:
    """
    Load leagues from JSON file or API and merge into the global LEAGUE_ALIASES dict.
    Returns the number of new aliases added.

    Priority:
      1. file_path argument or LEAGUES_JSON_FILE env var → load from local JSON
      2. url argument or LEAGUES_API_URL env var → fetch from API
      3. If both fail → keep existing static aliases

    Clears normalizer caches after updating so new aliases take effect.
    """
    # Try local JSON file first
    leagues = load_leagues_from_file(file_path)

    # Fall back to API if no file data
    if not leagues:
        leagues = fetch_leagues_from_api(url)

    if not leagues:
        logger.warning("No leagues loaded (file or API) — keeping existing static aliases")
        return 0

    dynamic_aliases = build_alias_map(leagues)

    # Merge: dynamic aliases are added, but static ones take priority
    added = 0
    for key, value in dynamic_aliases.items():
        if key not in LEAGUE_ALIASES:
            LEAGUE_ALIASES[key] = value
            added += 1

    # Clear caches so resolve_league_alias picks up new entries
    clear_caches()

    logger.info(
        f"League aliases updated: {added} new entries added "
        f"(total: {len(LEAGUE_ALIASES)})"
    )
    return added

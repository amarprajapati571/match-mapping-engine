"""
Dynamic Team Alias Loader — loads team data from JSON file or external API
and builds alias map for team name resolution.

Supports two data sources (priority order):
  1. Local JSON file  → set TEAMS_JSON_FILE=data/teams.json
  2. External API     → set TEAMS_API_URL=https://...

Each team document is expected to have:
  - team_name_en OR name_en: "Mumbai Indians"
  - team_abbreviation (optional): "MI"
  - keywords: ["MI", "Mumbai", "Mumbai Indians IPL"]

All variants (abbreviation, keywords) are mapped to the canonical team name
(lowercased, cleaned). This allows "MI" vs "Mumbai Indians" to resolve via
the alias dict even if acronym detection is not applicable.

Works alongside the acronym auto-detection in normalizer.py:
  - Dict lookup handles irregular abbreviations (e.g., "BVB" → "Borussia Dortmund")
  - Acronym detection handles first-letter abbreviations automatically
  - Together they cover nearly all abbreviation patterns without manual entries
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests

from config.settings import CONFIG
from core.normalizer import TEAM_ALIASES, clean_text, clear_caches

logger = logging.getLogger(__name__)


def _parse_json_data(data) -> list:
    """Extract team list from various JSON response formats."""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return (
            data.get("teams")
            or data.get("data")
            or data.get("rows")
            or data.get("results")
            or []
        )
    return []


def load_teams_from_file(file_path: str = None) -> list:
    """
    Load team data from a local JSON file.

    The JSON file can be:
      - A plain array:  [{"team_name_en": "...", ...}, ...]
      - A wrapped dict:  {"teams": [...]} or {"data": [...]}

    Returns raw list of team documents.
    """
    path = file_path or os.getenv("TEAMS_JSON_FILE", "")
    if not path:
        return []

    resolved = Path(path)
    if not resolved.is_absolute():
        # Resolve relative to project root
        resolved = Path(__file__).resolve().parent.parent / path

    if not resolved.exists():
        logger.warning(f"Teams JSON file not found: {resolved}")
        return []

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = _parse_json_data(data)
        logger.info(f"Loaded {len(teams)} teams from file: {resolved}")
        return teams
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read teams JSON file: {e}")
        return []


def fetch_teams_from_api(url: str = None, timeout: int = 30) -> list:
    """
    Fetch all teams from the external API.
    Supports optional auth via TEAMS_API_TOKEN or LEAGUES_API_TOKEN env var.
    Returns raw list of team documents.
    """
    api_url = url or CONFIG.endpoints.teams_api_url
    logger.info(f"Fetching teams from API: {api_url}")

    headers = {}
    token = os.getenv("TEAMS_API_TOKEN", "") or os.getenv("LEAGUES_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        teams = _parse_json_data(data)
        logger.info(f"Fetched {len(teams)} teams from API")
        return teams

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch teams from API: {e}")
        return []


def build_team_alias_map(teams: list) -> Dict[str, str]:
    """
    Build a team alias map from raw team documents.

    For each team, creates entries:
      abbreviation → canonical
      each keyword  → canonical

    Canonical form: cleaned team name (lowercased)

    Supports flexible field names:
      - team_name_en OR name_en → team name
      - team_abbreviation OR abbreviation → abbreviation (optional)
      - keywords → list of alternate names
    """
    alias_map = {}
    skipped = 0

    for team in teams:
        # Support both "team_name_en" and "name_en" field names
        team_name = (
            team.get("team_name_en", "")
            or team.get("name_en", "")
        ).strip()
        abbreviation = (
            team.get("team_abbreviation", "")
            or team.get("abbreviation", "")
        ).strip()
        keywords = team.get("keywords", [])

        if not team_name:
            skipped += 1
            continue

        canonical = clean_text(team_name)

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

    logger.info(
        f"Built {len(alias_map)} team aliases from {len(teams)} teams "
        f"({skipped} skipped)"
    )
    return alias_map


def load_team_aliases(url: str = None, file_path: str = None) -> int:
    """
    Load teams from JSON file or API and merge into the global TEAM_ALIASES dict.
    Returns the number of new aliases added.

    Priority:
      1. file_path argument or TEAMS_JSON_FILE env var → load from local JSON
      2. url argument or TEAMS_API_URL env var → fetch from API
      3. If both fail → keep existing static aliases + acronym detection

    Clears normalizer caches after updating so new aliases take effect.
    """
    # Try local JSON file first
    teams = load_teams_from_file(file_path)

    # Fall back to API if no file data
    if not teams:
        teams = fetch_teams_from_api(url)

    if not teams:
        logger.warning("No teams loaded (file or API) — keeping existing static aliases")
        return 0

    dynamic_aliases = build_team_alias_map(teams)

    # Merge: dynamic aliases are added, but static ones take priority
    added = 0
    for key, value in dynamic_aliases.items():
        if key not in TEAM_ALIASES:
            TEAM_ALIASES[key] = value
            added += 1

    # Clear caches so resolve_alias picks up new entries
    clear_caches()

    logger.info(
        f"Team aliases updated: {added} new entries added "
        f"(total: {len(TEAM_ALIASES)})"
    )
    return added

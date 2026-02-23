"""
Text normalization for sports match data.
Handles team aliases, category detection (Women/U23/Reserves/B-team),
and building clean text representations for embedding.

Performance: LRU caching on all expensive string operations eliminates
redundant computation for repeated team names across the B365/OP pools.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List, Tuple


# ═══════════════════════════════════════════════
# Category Detection Patterns
# ═══════════════════════════════════════════════

CATEGORY_PATTERNS = {
    "WOMEN": [
        r"\bwomen\b", r"\bwoman\b", r"\bfemale\b", r"\bfem\b",
        r"\bladies\b", r"\bw\b(?=\s|$)", r"\(w\)", r"\bwomens\b",
        r"\bfemenino\b", r"\bfeminino\b", r"\bdames\b", r"\bfrauen\b",
    ],
    "U23": [r"\bu-?23\b", r"\bunder[\s-]?23\b", r"\bsub[\s-]?23\b"],
    "U21": [r"\bu-?21\b", r"\bunder[\s-]?21\b", r"\bsub[\s-]?21\b"],
    "U20": [r"\bu-?20\b", r"\bunder[\s-]?20\b", r"\bsub[\s-]?20\b"],
    "U19": [r"\bu-?19\b", r"\bunder[\s-]?19\b", r"\bsub[\s-]?19\b"],
    "U18": [r"\bu-?18\b", r"\bunder[\s-]?18\b", r"\bsub[\s-]?18\b"],
    "U17": [r"\bu-?17\b", r"\bunder[\s-]?17\b", r"\bsub[\s-]?17\b"],
    "RESERVES": [
        r"\breserves?\b", r"\breserva\b", r"\bii\b",
        r"\b2nd\b", r"\bsecond\s+team\b",
    ],
    "B-TEAM": [
        r"\bb[\s-]?team\b", r"\b[b]\b(?=\s|$)", r"\batlético\s+b\b",
        r"\bcastilla\b", r"\bfabril\b",
    ],
    "YOUTH": [r"\byouth\b", r"\bjunior\b", r"\bjuvenil\b", r"\bjugend\b"],
    "AMATEUR": [r"\bamateur\b", r"\bamador\b"],
}


# ═══════════════════════════════════════════════
# Common Team Aliases (expandable)
# ═══════════════════════════════════════════════

TEAM_ALIASES = {
    # English
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "newcastle utd": "newcastle united",
    "tottenham": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west ham": "west ham united",
    "brighton": "brighton and hove albion",
    "nottm forest": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "sheffield utd": "sheffield united",
    "leeds utd": "leeds united",

    # European
    "atletico madrid": "atletico de madrid",
    "atletico": "atletico de madrid",
    "real sociedad": "real sociedad de futbol",
    "bayern": "bayern munich",
    "fc bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "bvb": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "psg": "paris saint-germain",
    "paris sg": "paris saint-germain",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "ac milan": "milan",
    "napoli": "ssc napoli",
    "juve": "juventus",

    # South American
    "boca": "boca juniors",
    "river": "river plate",
    "flamengo": "cr flamengo",
    "palmeiras": "se palmeiras",
    "corinthians": "sc corinthians",

    # International short forms
    "usa": "united states",
    "uae": "united arab emirates",
    "south korea": "korea republic",
    "s. korea": "korea republic",
    "north korea": "korea dpr",
}

_COMPILED_CATEGORY_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in CATEGORY_PATTERNS.items()
}

_STRIP_PATTERN = re.compile(r"[^\w\s|.\-]")
_MULTI_SPACE = re.compile(r"\s+")
_SPLIT_PATTERN = re.compile(r"[\s\-/]+")

_NOISE_TOKENS = frozenset({
    "fc", "sc", "cf", "ac", "afc", "ssc", "se", "cr", "fk", "sk",
    "bk", "if", "hc", "ik", "the", "de", "la", "el", "al",
})


# ═══════════════════════════════════════════════
# Cached core functions
# ═══════════════════════════════════════════════

@lru_cache(maxsize=16384)
def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (accents, special chars)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=16384)
def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip extra whitespace, normalize."""
    text = normalize_unicode(text)
    text = text.lower().strip()
    text = _STRIP_PATTERN.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


@lru_cache(maxsize=8192)
def resolve_alias(team_name: str) -> str:
    """Resolve common team aliases to canonical names."""
    cleaned = clean_text(team_name)
    return TEAM_ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=4096)
def _detect_categories_cached(text_lower: str) -> tuple:
    """Cached category detection returning immutable tuple."""
    categories = []
    for cat, patterns in _COMPILED_CATEGORY_PATTERNS.items():
        for p in patterns:
            if p.search(text_lower):
                categories.append(cat)
                break
    return tuple(categories)


def detect_categories(text: str) -> List[str]:
    """Detect category tags (WOMEN, U23, RESERVES, etc.) from text."""
    return list(_detect_categories_cached(text.lower()))


def strip_category_tokens(text: str) -> str:
    """Remove category indicators from text for cleaner matching."""
    for patterns in CATEGORY_PATTERNS.values():
        for p in patterns:
            text = re.sub(p, " ", text, flags=re.IGNORECASE)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=16384)
def _build_match_text_cached(
    league: str,
    home_team: str,
    away_team: str,
    tags_tuple: tuple,
    include_tags: bool,
) -> str:
    home_clean = resolve_alias(home_team)
    away_clean = resolve_alias(away_team)
    league_clean = clean_text(league)

    tag_prefix = ""
    if include_tags and tags_tuple:
        tag_prefix = " ".join(f"[{t}]" for t in sorted(tags_tuple)) + " "

    return f"{tag_prefix}{league_clean} | {home_clean} vs {away_clean}"


def build_match_text(
    league: str,
    home_team: str,
    away_team: str,
    category_tags=None,
    include_tags: bool = True,
) -> str:
    """
    Build the text representation used for embedding.
    Format: [TAG1] [TAG2] League | Home vs Away

    Accepts list or tuple for category_tags (converted to tuple for caching).
    """
    tags = tuple(category_tags) if category_tags else ()
    return _build_match_text_cached(league, home_team, away_team, tags, include_tags)


def build_swapped_text(
    league: str,
    home_team: str,
    away_team: str,
    category_tags=None,
    include_tags: bool = True,
) -> str:
    """Build text with home/away swapped."""
    tags = tuple(category_tags) if category_tags else ()
    return _build_match_text_cached(league, away_team, home_team, tags, include_tags)


# ═══════════════════════════════════════════════
# Team Name Similarity (heavily cached)
# ═══════════════════════════════════════════════

@lru_cache(maxsize=8192)
def _tokenize(name: str) -> frozenset:
    """Normalize a team name to a frozenset of lowercase tokens."""
    cleaned = resolve_alias(name)
    parts = _SPLIT_PATTERN.split(cleaned)
    return frozenset(t for t in parts if t and t not in _NOISE_TOKENS)


@lru_cache(maxsize=32768)
def _team_pair_sim(name_a: str, name_b: str) -> float:
    """
    Similarity between two team names (0-1).
    Combines token-overlap (Jaccard) and character-level sequence matching.
    """
    tokens_a = _tokenize(name_a)
    tokens_b = _tokenize(name_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)

    str_a = resolve_alias(name_a)
    str_b = resolve_alias(name_b)
    seq_ratio = SequenceMatcher(None, str_a, str_b).ratio()

    return 0.6 * jaccard + 0.4 * seq_ratio


def _combine_pair_scores(sim_a: float, sim_b: float) -> float:
    """
    Combine two team-pair similarities into one match-level score.
    Both teams must contribute; one weak match doesn't zero everything.
    """
    if sim_a == 0.0 and sim_b == 0.0:
        return 0.0
    return 0.4 * min(sim_a, sim_b) + 0.6 * (sim_a * sim_b) ** 0.5


@lru_cache(maxsize=65536)
def compute_team_similarity(
    op_home: str, op_away: str,
    b365_home: str, b365_away: str,
) -> Tuple[float, bool]:
    """
    Compute best team-pair similarity between an OP match and a B365 match.
    Tries both normal and swapped orientation.

    Returns (best_similarity 0-1, is_swapped).
    """
    home_sim = _team_pair_sim(op_home, b365_home)
    away_sim = _team_pair_sim(op_away, b365_away)
    normal_score = _combine_pair_scores(home_sim, away_sim)

    swap_home_sim = _team_pair_sim(op_home, b365_away)
    swap_away_sim = _team_pair_sim(op_away, b365_home)
    swapped_score = _combine_pair_scores(swap_home_sim, swap_away_sim)

    if swapped_score > normal_score:
        return swapped_score, True
    return normal_score, False


def extract_all_info(raw_match: dict) -> Tuple[str, str, List[str]]:
    """
    From a raw match dict, extract normalized text + swapped text + categories.
    Returns: (normal_text, swapped_text, category_tags)
    """
    home = raw_match.get("home_team", "")
    away = raw_match.get("away_team", "")
    league = raw_match.get("league", "")

    full_text = f"{league} {home} {away}"
    cats = detect_categories(full_text)

    explicit_cats = raw_match.get("category_tags", [])
    all_cats = tuple(sorted(set(cats + explicit_cats)))

    normal = build_match_text(league, home, away, all_cats)
    swapped = build_swapped_text(league, home, away, all_cats)

    return normal, swapped, list(all_cats)


def clear_caches():
    """Clear all LRU caches (call after alias table updates)."""
    normalize_unicode.cache_clear()
    clean_text.cache_clear()
    resolve_alias.cache_clear()
    _detect_categories_cached.cache_clear()
    _build_match_text_cached.cache_clear()
    _tokenize.cache_clear()
    _team_pair_sim.cache_clear()
    compute_team_similarity.cache_clear()

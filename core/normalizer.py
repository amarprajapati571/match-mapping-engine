"""
Text normalization for sports match data.
Handles team aliases, category detection (Women/U23/Reserves/B-team),
and building clean text representations for embedding.

Intelligent Abbreviation Resolution (3-layer approach):
  1. Static alias dicts  — for nicknames and irregular abbreviations
  2. Dynamic API loading  — fetches aliases from external team/league databases
  3. Acronym auto-detect  — algorithmically detects first-letter abbreviations
     e.g., "MI" ↔ "Mumbai Indians", "IPL" ↔ "Indian Premier League"
     No dictionary entry needed — works for ANY language/sport.

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

    # Cricket — IPL
    "mi": "mumbai indians",
    "csk": "chennai super kings",
    "rcb": "royal challengers bangalore",
    "royal challengers bengaluru": "royal challengers bangalore",
    "kkr": "kolkata knight riders",
    "srh": "sunrisers hyderabad",
    "dc": "delhi capitals",
    "dd": "delhi daredevils",
    "rr": "rajasthan royals",
    "pbks": "punjab kings",
    "kxip": "kings xi punjab",
    "kings xi punjab": "punjab kings",
    "gt": "gujarat titans",
    "lsg": "lucknow super giants",

    # Cricket — International
    "ind": "india",
    "aus": "australia",
    "eng": "england",
    "pak": "pakistan",
    "sa": "south africa",
    "nz": "new zealand",
    "wi": "west indies",
    "sl": "sri lanka",
    "ban": "bangladesh",
    "afg": "afghanistan",
    "zim": "zimbabwe",
    "ire": "ireland",

    # Cricket — Other T20 leagues
    "tkr": "trinbago knight riders",
    "bpb": "barbados royals",
    "ms": "melbourne stars",
    "mr": "melbourne renegades",
    "ss": "sydney sixers",
    "st": "sydney thunder",
    "ps": "perth scorchers",
    "bh": "brisbane heat",
    "as": "adelaide strikers",
    "hh": "hobart hurricanes",

    # Basketball — NBA
    "lal": "los angeles lakers",
    "la lakers": "los angeles lakers",
    "lakers": "los angeles lakers",
    "lac": "los angeles clippers",
    "la clippers": "los angeles clippers",
    "clippers": "los angeles clippers",
    "gsw": "golden state warriors",
    "warriors": "golden state warriors",
    "bos": "boston celtics",
    "celtics": "boston celtics",
    "nyk": "new york knicks",
    "knicks": "new york knicks",
    "bkn": "brooklyn nets",
    "nets": "brooklyn nets",
    "chi": "chicago bulls",
    "bulls": "chicago bulls",
    "mia": "miami heat",
    "heat": "miami heat",
    "dal": "dallas mavericks",
    "mavs": "dallas mavericks",
    "mavericks": "dallas mavericks",
    "den": "denver nuggets",
    "nuggets": "denver nuggets",
    "mil": "milwaukee bucks",
    "bucks": "milwaukee bucks",
    "phi": "philadelphia 76ers",
    "sixers": "philadelphia 76ers",
    "phx": "phoenix suns",
    "suns": "phoenix suns",
}

# ═══════════════════════════════════════════════
# Common League Aliases (abbreviations → full names)
# ═══════════════════════════════════════════════

LEAGUE_ALIASES = {
    # Cricket
    "ipl": "indian premier league",
    "bpl": "bangladesh premier league",
    "bbl": "big bash league",
    "cpl": "caribbean premier league",
    "psl": "pakistan super league",
    "t20": "twenty20",

    # Football / Soccer
    "epl": "english premier league",
    "pl": "premier league",
    "ucl": "uefa champions league",
    "uel": "uefa europa league",
    "uecl": "uefa europa conference league",
    "mls": "major league soccer",
    "a-league": "australian a-league",
    "j-league": "japanese j league",
    "k-league": "korean k league",
    "csl": "chinese super league",
    "isl": "indian super league",
    "spfl": "scottish premiership",
    "eredivisie": "dutch eredivisie",
    "ligue 1": "french ligue 1",
    "serie a": "italian serie a",
    "la liga": "spanish la liga",
    "bundesliga": "german bundesliga",
    "liga mx": "mexican liga mx",
    "copa lib": "copa libertadores",
    "copa sudo": "copa sudamericana",
    "afcon": "africa cup of nations",
    "concacaf cl": "concacaf champions league",

    # Basketball
    "nba": "national basketball association",
    "wnba": "womens national basketball association",
    "euroleague": "turkish airlines euroleague",
    "ncaa": "national collegiate athletic association",
    "nbl": "national basketball league",

    # American Football
    "nfl": "national football league",
    "cfl": "canadian football league",
    "xfl": "xfl",

    # Baseball
    "mlb": "major league baseball",
    "npb": "nippon professional baseball",
    "kbo": "korean baseball organization",

    # Hockey
    "nhl": "national hockey league",
    "khl": "kontinental hockey league",
    "shl": "swedish hockey league",

    # Tennis
    "atp": "association of tennis professionals",
    "wta": "womens tennis association",

    # Rugby
    "nrl": "national rugby league",
    "urc": "united rugby championship",
    "super rugby": "super rugby pacific",

    # Esports
    "lec": "league of legends european championship",
    "lck": "league of legends champions korea",
    "lpl": "league of legends pro league",
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
# Acronym Auto-Detection
# ═══════════════════════════════════════════════
# Handles first-letter abbreviations automatically without any dictionary.
# e.g., "MI" ↔ "Mumbai Indians", "CSK" ↔ "Chennai Super Kings",
#        "IPL" ↔ "Indian Premier League", "UEFA" ↔ "Union of European Football Assoc."

_ACRONYM_STOP_WORDS = frozenset({
    "of", "the", "and", "de", "la", "el", "von", "van", "le", "les",
    "di", "da", "do", "das", "del", "des", "du", "en", "et", "y",
})

# Matches single-letter tokens with optional period: "n.", "r.", "a"
# These are suffix initials in names like "Djokovic N." or "Nadal R."
_INITIAL_PATTERN = re.compile(r'^[a-z]\.?$')


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
def resolve_league_alias(league_name: str) -> str:
    """Resolve common league abbreviations to canonical full names."""
    cleaned = clean_text(league_name)
    return LEAGUE_ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=8192)
def is_acronym_of(short_text: str, long_text: str) -> bool:
    """
    Check if short_text is a first-letter acronym of long_text.

    This replaces the need for manual alias entries for standard abbreviations.
    Stop words (of, the, and, de, ...) are skipped in the long text.

    Examples:
        is_acronym_of("MI", "Mumbai Indians")           → True  (M=M, I=I)
        is_acronym_of("CSK", "Chennai Super Kings")      → True  (C=C, S=S, K=K)
        is_acronym_of("IPL", "Indian Premier League")    → True  (I=I, P=P, L=L)
        is_acronym_of("UEFA", "Union of European Football Associations") → True
        is_acronym_of("PSG", "Paris Saint-Germain")      → True  (P=P, S=S, G=G)
        is_acronym_of("BVB", "Borussia Dortmund")        → False (B≠D)
    """
    # Clean and strip dots/hyphens/spaces from the short text
    short = re.sub(r'[.\-\s]', '', clean_text(short_text))
    if len(short) < 2:
        return False

    # Split long text into words, skip stop words
    long_clean = clean_text(long_text)
    words = _SPLIT_PATTERN.split(long_clean)
    words = [w for w in words if w and w not in _ACRONYM_STOP_WORDS]

    if not words or len(short) != len(words):
        return False

    return all(s == w[0] for s, w in zip(short, words))


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
    for compiled_patterns in _COMPILED_CATEGORY_PATTERNS.values():
        for p in compiled_patterns:
            text = p.sub(" ", text)
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
    """
    Normalize a team name to a frozenset of lowercase tokens.
    Strips noise tokens (FC, SC, etc.) AND suffix initials (N., R., etc.)
    so "Djokovic N." → {"djokovic"} and "Nadal R." → {"nadal"}.
    """
    cleaned = resolve_alias(name)
    parts = _SPLIT_PATTERN.split(cleaned)
    return frozenset(
        t for t in parts
        if t and t not in _NOISE_TOKENS and not _INITIAL_PATTERN.match(t)
    )


@lru_cache(maxsize=32768)
def _team_pair_sim(name_a: str, name_b: str) -> float:
    """
    Similarity between two team names (0-1).

    Resolution order:
      1. Alias dict lookup (handles nicknames: "Spurs" → "Tottenham Hotspur")
      2. Acronym auto-detection (handles "MI" ↔ "Mumbai Indians" without dict)
      3. Token overlap (Jaccard) + character-level sequence matching (SequenceMatcher)
    """
    str_a = resolve_alias(name_a)
    str_b = resolve_alias(name_b)

    # Layer 2: Acronym detection — no dictionary needed
    # Catches "MI" vs "Mumbai Indians", "CSK" vs "Chennai Super Kings", etc.
    if is_acronym_of(str_a, str_b) or is_acronym_of(str_b, str_a):
        return 1.0

    # Layer 3: Token overlap + character-level matching
    tokens_a = _tokenize(name_a)
    tokens_b = _tokenize(name_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)

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


@lru_cache(maxsize=8192)
def compute_league_similarity(league_a: str, league_b: str) -> float:
    """
    Compute similarity between two league names (0-1).

    Resolution order:
      1. Alias dict lookup (handles known abbreviations)
      2. Acronym auto-detection (handles any first-letter abbreviation)
      3. Token overlap (Jaccard) + character sequence matching
    """
    a = resolve_league_alias(league_a)
    b = resolve_league_alias(league_b)

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Layer 2: Acronym detection — catches any abbreviation not in the dict
    # e.g., "BBL" vs "Big Bash League", "SA20" edge cases, etc.
    if is_acronym_of(a, b) or is_acronym_of(b, a):
        return 1.0

    # Layer 3: Token overlap + character-level matching
    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if tokens_a and tokens_b:
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union)
    else:
        jaccard = 0.0

    seq_ratio = SequenceMatcher(None, a, b).ratio()

    return 0.4 * jaccard + 0.6 * seq_ratio


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
    resolve_league_alias.cache_clear()
    is_acronym_of.cache_clear()
    _detect_categories_cached.cache_clear()
    _build_match_text_cached.cache_clear()
    _tokenize.cache_clear()
    _team_pair_sim.cache_clear()
    compute_team_similarity.cache_clear()
    compute_league_similarity.cache_clear()

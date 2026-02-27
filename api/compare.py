"""
Match Compare — Direct pairwise comparison between a provider match and a Bet365 match.

Computes all similarity metrics, gate evaluation, and swap detection
WITHOUT requiring the B365 pool to be indexed. Uses the cross-encoder
model directly for scoring.
"""

from datetime import datetime
from typing import Any, List

import numpy as np
from pydantic import BaseModel

from config.settings import CONFIG
from core.normalizer import (
    build_match_text,
    build_swapped_text,
    compute_league_similarity,
    compute_team_similarity,
    detect_categories,
)
from core.inference import _sigmoid_np


# ═══════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════


class MatchInput(BaseModel):
    home_team: str
    away_team: str
    league: str
    sport: str = "soccer"
    kickoff: datetime


class CompareRequest(BaseModel):
    provider_match: MatchInput
    bet365_match: MatchInput


class GateDetail(BaseModel):
    name: str
    passed: bool
    actual_value: Any
    threshold: Any
    description: str


class CompareResponse(BaseModel):
    # Core scores
    confidence_score: float
    cross_encoder_raw: float
    cross_encoder_normalized: float
    normal_ce_score: float
    swapped_ce_score: float

    # Similarity
    team_similarity: float
    league_similarity: float

    # Sport
    provider_sport: str
    bet365_sport: str
    sport_match: bool

    # Swap
    is_swapped: bool
    swap_reason: str

    # Time
    time_diff_minutes: float
    time_diff_hours: float

    # Categories
    provider_categories: List[str]
    bet365_categories: List[str]
    categories_match: bool

    # Gates
    gates: List[GateDetail]
    all_gates_pass: bool

    # Verdict
    verdict: str
    verdict_reason: str

    # Debug text
    provider_text: str
    bet365_text: str
    provider_swapped_text: str


# ═══════════════════════════════════════════════
# Comparison Logic
# ═══════════════════════════════════════════════


def compare_matches(provider: MatchInput, bet365: MatchInput, engine) -> CompareResponse:
    """
    Perform a full pairwise comparison between a provider match and a Bet365 match.

    Returns all metrics: confidence, swap detection, team/league similarity,
    time difference, gate evaluation, and overall verdict.
    """
    gates_cfg = CONFIG.gates

    # ── Step 1: Category detection ──
    provider_full = f"{provider.league} {provider.home_team} {provider.away_team}"
    bet365_full = f"{bet365.league} {bet365.home_team} {bet365.away_team}"
    op_cats = detect_categories(provider_full)
    b365_cats = detect_categories(bet365_full)

    # ── Step 2: Build text representations ──
    op_text = build_match_text(
        provider.league, provider.home_team, provider.away_team, op_cats,
    )
    b365_text = build_match_text(
        bet365.league, bet365.home_team, bet365.away_team, b365_cats,
    )
    op_swapped_text = build_swapped_text(
        provider.league, provider.home_team, provider.away_team, op_cats,
    )

    # ── Step 3: Cross-encoder scoring (both orientations) ──
    pairs = [(op_text, b365_text), (op_swapped_text, b365_text)]
    scores = engine.cross_encoder.predict(pairs)
    normal_ce = float(scores[0])
    swapped_ce = float(scores[1])

    # ── Step 4: Sigmoid normalization ──
    best_ce_raw = max(normal_ce, swapped_ce)
    norm_ce = float(_sigmoid_np(np.array([best_ce_raw]))[0])

    # ── Step 5: Team similarity ──
    team_sim, sim_swapped = compute_team_similarity(
        provider.home_team, provider.away_team,
        bet365.home_team, bet365.away_team,
    )

    # ── Step 6: League similarity ──
    league_sim = compute_league_similarity(provider.league, bet365.league)

    # ── Step 7: Time difference ──
    time_diff_sec = abs((provider.kickoff - bet365.kickoff).total_seconds())
    time_diff_min = time_diff_sec / 60.0
    time_diff_hr = time_diff_min / 60.0

    # ── Step 7b: Sport match check ──
    sport_match = provider.sport.strip().lower() == bet365.sport.strip().lower()

    # ── Step 8: Final weighted score ──
    # Priority: sport, team_name, kickoff = TOP (hard gates, zero on mismatch)
    #           league = LOW (soft factor, reduces score but never zeros it)
    w = gates_cfg.team_sim_weight
    min_tsim = gates_cfg.min_team_similarity
    max_kickoff = gates_cfg.max_kickoff_diff_minutes
    league_w = gates_cfg.league_soft_weight  # max penalty for league mismatch

    if not sport_match:
        final_score = 0.0
    elif team_sim < min_tsim:
        final_score = 0.0
    elif time_diff_min > max_kickoff:
        final_score = 0.0
    else:
        base_score = (1 - w) * norm_ce + w * team_sim
        # League as soft multiplier: (1 - league_w) to 1.0
        # E.g., league_w=0.15 → factor ranges from 0.85 (no match) to 1.0 (perfect)
        league_factor = (1.0 - league_w) + league_w * league_sim
        final_score = base_score * league_factor

    # ── Step 9: Swap detection ──
    ce_says_swapped = swapped_ce > normal_ce
    is_swapped = sim_swapped if team_sim >= min_tsim else ce_says_swapped

    swap_reason = "none"
    if is_swapped:
        swap_reason = "team_similarity" if sim_swapped else "cross_encoder"

    # ── Step 10: Gate evaluation ──
    gate_results = []

    # 0. sport_gate (must be same sport)
    gate_results.append(GateDetail(
        name="sport_gate",
        passed=sport_match,
        actual_value=f"{provider.sport} vs {bet365.sport}",
        threshold="exact match",
        description=f"Sport {'matches' if sport_match else 'MISMATCH'}: {provider.sport} vs {bet365.sport}",
    ))

    # 1. min_score_gate
    gate_results.append(GateDetail(
        name="min_score_gate",
        passed=final_score >= gates_cfg.min_score,
        actual_value=round(final_score, 4),
        threshold=gates_cfg.min_score,
        description=f"Score {final_score:.4f} {'≥' if final_score >= gates_cfg.min_score else '<'} {gates_cfg.min_score}",
    ))

    # 2. margin_gate (N/A for single pair — always pass)
    gate_results.append(GateDetail(
        name="margin_gate",
        passed=True,
        actual_value="N/A",
        threshold=gates_cfg.margin,
        description="Single-pair comparison (no second candidate)",
    ))

    # 3. category_gate
    sensitive = set(gates_cfg.sensitive_categories)
    all_cats_union = set(op_cats) | set(b365_cats)
    has_sensitive = bool(all_cats_union & sensitive)

    if gates_cfg.block_sensitive_auto_match and has_sensitive:
        cat_pass = False
        cat_desc = f"Sensitive category detected: {all_cats_union & sensitive}"
    elif set(op_cats) == set(b365_cats):
        cat_pass = True
        cat_desc = "Categories match"
    else:
        cat_pass = False
        cat_desc = f"Mismatch: {op_cats} vs {b365_cats}"

    gate_results.append(GateDetail(
        name="category_gate",
        passed=cat_pass,
        actual_value=f"{op_cats} vs {b365_cats}",
        threshold="exact match",
        description=cat_desc,
    ))

    # 4. kickoff_gate (±45 minutes hard cutoff)
    kickoff_pass = time_diff_min <= max_kickoff
    gate_results.append(GateDetail(
        name="kickoff_gate",
        passed=kickoff_pass,
        actual_value=round(time_diff_min, 1),
        threshold=max_kickoff,
        description=f"{time_diff_min:.1f}min {'≤' if kickoff_pass else '>'} {max_kickoff}min",
    ))

    # 5. team_name_gate
    team_pass = team_sim >= gates_cfg.min_team_similarity
    gate_results.append(GateDetail(
        name="team_name_gate",
        passed=team_pass,
        actual_value=round(team_sim, 4),
        threshold=gates_cfg.min_team_similarity,
        description=f"Team similarity {team_sim:.4f} {'≥' if team_pass else '<'} {gates_cfg.min_team_similarity}",
    ))

    # 6. league_gate (SOFT — informational only, does NOT block verdict)
    league_pass = league_sim >= gates_cfg.min_league_similarity
    league_factor_val = (1.0 - gates_cfg.league_soft_weight) + gates_cfg.league_soft_weight * league_sim
    gate_results.append(GateDetail(
        name="league_gate",
        passed=league_pass,
        actual_value=round(league_sim, 4),
        threshold=f"{gates_cfg.min_league_similarity} (soft — penalty factor: {league_factor_val:.2f})",
        description=(
            f"League similarity {league_sim:.4f} — "
            f"{'OK' if league_pass else 'LOW (soft penalty applied)'} "
            f"(factor: {league_factor_val:.2f}×)"
        ),
    ))

    # Verdict uses only hard gates (sport, score, margin, category, kickoff, team)
    # League is excluded — it's a soft factor that reduces score, never blocks
    hard_gates = [g for g in gate_results if g.name != "league_gate"]
    all_pass = all(g.passed for g in hard_gates)

    # ── Step 11: Verdict ──
    if all_pass and is_swapped:
        verdict = "SWAPPED_MATCH"
        verdict_reason = "All gates pass. Teams are swapped between providers."
    elif all_pass:
        verdict = "MATCH"
        verdict_reason = "All gates pass. High confidence match."
    else:
        failed = [g.name for g in gate_results if not g.passed]
        verdict = "NO_MATCH"
        verdict_reason = f"Failed gates: {', '.join(failed)}"

    return CompareResponse(
        confidence_score=round(final_score, 4),
        cross_encoder_raw=round(best_ce_raw, 4),
        cross_encoder_normalized=round(norm_ce, 4),
        normal_ce_score=round(normal_ce, 4),
        swapped_ce_score=round(swapped_ce, 4),
        team_similarity=round(team_sim, 4),
        league_similarity=round(league_sim, 4),
        provider_sport=provider.sport,
        bet365_sport=bet365.sport,
        sport_match=sport_match,
        is_swapped=is_swapped,
        swap_reason=swap_reason,
        time_diff_minutes=round(time_diff_min, 1),
        time_diff_hours=round(time_diff_hr, 2),
        provider_categories=op_cats,
        bet365_categories=b365_cats,
        categories_match=set(op_cats) == set(b365_cats),
        gates=gate_results,
        all_gates_pass=all_pass,
        verdict=verdict,
        verdict_reason=verdict_reason,
        provider_text=op_text,
        bet365_text=b365_text,
        provider_swapped_text=op_swapped_text,
    )

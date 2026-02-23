"""
Data models / schemas for the Match Mapping Engine.
Pydantic models for API I/O and internal state.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


# ═══════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════

class Decision(str, Enum):
    MATCH = "MATCH"
    SWAPPED = "SWAPPED"
    NO_MATCH = "NO_MATCH"


class CSEFeedback(str, Enum):
    """Feedback types from the CSE team review."""
    CORRECT = "Correct"
    NOT_SURE = "Not Sure"
    NOT_CORRECT = "Not correct"
    NEED_TO_SWAP = "Need to swap"

    @staticmethod
    def to_decision(feedback: "CSEFeedback") -> Optional[Decision]:
        """Map CSE feedback to internal training decision.
        Returns None for NOT_SURE (skip training on uncertain data).
        """
        mapping = {
            CSEFeedback.CORRECT: Decision.MATCH,
            CSEFeedback.NOT_CORRECT: Decision.NO_MATCH,
            CSEFeedback.NEED_TO_SWAP: Decision.SWAPPED,
        }
        return mapping.get(feedback)


class GateResult(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    NEED_REVIEW = "NEED_REVIEW"


class CategoryTag(str, Enum):
    WOMEN = "WOMEN"
    U23 = "U23"
    U21 = "U21"
    U20 = "U20"
    U19 = "U19"
    U18 = "U18"
    U17 = "U17"
    RESERVES = "RESERVES"
    B_TEAM = "B-TEAM"
    YOUTH = "YOUTH"
    AMATEUR = "AMATEUR"
    NONE = "NONE"


# ═══════════════════════════════════════════════
# Match Data
# ═══════════════════════════════════════════════

class MatchRecord(BaseModel):
    """A match from either platform (OP or B365)."""
    match_id: str
    platform: str                    # "OP" or "B365"
    sport: str                       # e.g., "soccer", "basketball"
    league: str                      # e.g., "Premier League"
    home_team: str
    away_team: str
    kickoff: datetime
    category_tags: List[str] = []    # ["WOMEN"], ["U23"], etc.
    raw_text: Optional[str] = None   # Pre-built text repr for embedding

    def build_text(self) -> str:
        """Build text representation using cached normalizer (alias resolution + LRU cache)."""
        from core.normalizer import build_match_text
        text = build_match_text(
            self.league, self.home_team, self.away_team, self.category_tags,
        )
        self.raw_text = text
        return text


# ═══════════════════════════════════════════════
# Inference Output
# ═══════════════════════════════════════════════

class Candidate(BaseModel):
    """A single B365 candidate for an OP match."""
    rank: int
    b365_match_id: str
    b365_home: str
    b365_away: str
    b365_kickoff: datetime
    score: float                     # Final reranked score (0-1)
    time_diff_minutes: float
    swapped: bool = False            # True if home/away were swapped to get best score
    category_tags: List[str] = []


class MappingSuggestion(BaseModel):
    """Full inference output for one OP match."""
    mapping_suggestion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    op_match_id: str
    op_home: str
    op_away: str
    op_sport: str
    op_league: str
    op_kickoff: datetime
    candidates_top5: List[Candidate]
    gate_decision: GateResult
    gate_details: dict = {}          # Which gates passed/failed
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════
# Feedback / Human Labels
# ═══════════════════════════════════════════════

class FeedbackRecord(BaseModel):
    """Human decision from the Admin UI."""
    mapping_suggestion_id: str
    op_match_id: str
    decision: Decision
    selected_b365_match_id: Optional[str] = None   # None if NO_MATCH
    swapped: bool = False
    reason_code: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


class CSEFeedbackRecord(BaseModel):
    """A single feedback row returned by the CSE feedback API."""
    provider_id: str
    bet365_match: Optional[str] = None
    platform: str = "ODDSPORTAL"
    feedback: CSEFeedback
    confidence: Optional[float] = None
    switch: bool = False
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    bet365_home_team: Optional[str] = None
    bet365_away_team: Optional[str] = None
    league: Optional[str] = None
    bet365_league: Optional[str] = None
    sport: Optional[str] = None


# ═══════════════════════════════════════════════
# Training Data
# ═══════════════════════════════════════════════

class TrainingPair(BaseModel):
    """A single training record for SBERT / Cross-Encoder."""
    pair_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anchor_text: str         # OP match text
    candidate_text: str      # B365 match text
    label: float             # 1.0 = positive (match/swapped), 0.0 = negative
    is_hard_negative: bool = False
    source_suggestion_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

class EvalResult(BaseModel):
    """Evaluation metrics snapshot."""
    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    dataset_size: int
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    auto_match_precision: float
    auto_match_rate: float
    no_match_false_positive_rate: float
    per_sport_metrics: dict = {}
    per_category_metrics: dict = {}
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

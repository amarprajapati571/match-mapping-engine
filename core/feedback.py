"""
Feedback Ingestion Pipeline.

Consumes human decisions from Admin UI and converts them into training data.
- MATCH / SWAPPED → positive pairs
- Unselected Top-5 candidates → hard negatives
- NO_MATCH → all candidates become hard negatives

Enforces 1:1 mapping constraint to prevent duplicate confirmations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.models import (
    Decision, FeedbackRecord, MappingSuggestion,
    TrainingPair, MatchRecord, CSEFeedback, CSEFeedbackRecord,
)
from core.normalizer import build_match_text, build_swapped_text
from config.settings import CONFIG

logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    In-memory feedback store with 1:1 mapping enforcement.
    Production: replace with MongoDB/PostgreSQL backend.
    """
    
    def __init__(self):
        # feedback_id → FeedbackRecord
        self._feedbacks: Dict[str, FeedbackRecord] = {}
        # suggestion_id → MappingSuggestion (cached for training data generation)
        self._suggestions: Dict[str, MappingSuggestion] = {}
        # op_match_id → confirmed b365_match_id (1:1 constraint)
        self._op_to_b365: Dict[str, str] = {}
        # b365_match_id → confirmed op_match_id (reverse 1:1)
        self._b365_to_op: Dict[str, str] = {}
        # Training pairs (all, plus indexed sublists for O(1) access)
        self._training_pairs: List[TrainingPair] = []
        self._positives: List[TrainingPair] = []
        self._hard_negatives: List[TrainingPair] = []
    
    def store_suggestion(self, suggestion: MappingSuggestion):
        """Cache a suggestion for later feedback processing."""
        self._suggestions[suggestion.mapping_suggestion_id] = suggestion
    
    def ingest_feedback(
        self,
        feedback: FeedbackRecord,
        op_record: Optional[MatchRecord] = None,
        b365_records: Optional[Dict[str, MatchRecord]] = None,
    ) -> Tuple[bool, str]:
        """
        Process a human decision.
        
        Args:
            feedback: The human decision
            op_record: OP match record (for text building)
            b365_records: Dict of b365_match_id → MatchRecord
            
        Returns:
            (success, message)
        """
        sid = feedback.mapping_suggestion_id
        
        # Validate suggestion exists
        suggestion = self._suggestions.get(sid)
        if not suggestion:
            return False, f"Unknown suggestion_id: {sid}"
        
        # ── 1:1 Constraint Check ──
        if feedback.decision in (Decision.MATCH, Decision.SWAPPED):
            b365_id = feedback.selected_b365_match_id
            if not b365_id:
                return False, "MATCH/SWAPPED requires selected_b365_match_id"
            
            # Check OP already mapped
            existing_b365 = self._op_to_b365.get(feedback.op_match_id)
            if existing_b365 and existing_b365 != b365_id:
                return False, (
                    f"OP {feedback.op_match_id} already mapped to "
                    f"B365 {existing_b365}. Duplicate mapping rejected."
                )
            
            # Check B365 already mapped to different OP
            existing_op = self._b365_to_op.get(b365_id)
            if existing_op and existing_op != feedback.op_match_id:
                return False, (
                    f"B365 {b365_id} already mapped to OP {existing_op}. "
                    f"1:1 constraint violated."
                )
            
            # Register mapping
            self._op_to_b365[feedback.op_match_id] = b365_id
            self._b365_to_op[b365_id] = feedback.op_match_id
        
        # Store feedback
        self._feedbacks[sid] = feedback
        
        # ── Generate Training Pairs ──
        pairs = self._generate_training_pairs(
            feedback, suggestion, op_record, b365_records
        )
        self._training_pairs.extend(pairs)
        for p in pairs:
            if p.label == 1.0:
                self._positives.append(p)
            if p.is_hard_negative:
                self._hard_negatives.append(p)
        
        logger.info(
            f"Feedback ingested: {feedback.decision.value} for "
            f"suggestion {sid} → {len(pairs)} training pairs"
        )
        
        return True, f"OK: {len(pairs)} training pairs generated"
    
    def _generate_training_pairs(
        self,
        feedback: FeedbackRecord,
        suggestion: MappingSuggestion,
        op_record: Optional[MatchRecord],
        b365_records: Optional[Dict[str, MatchRecord]],
    ) -> List[TrainingPair]:
        """
        Convert a single feedback into training pairs.
        
        Rules:
        - MATCH/SWAPPED: confirmed pair = positive, unselected = hard negatives
        - NO_MATCH: all candidates = hard negatives
        - NEVER include true positive in negative pool (label contamination prevention)
        """
        pairs = []
        sid = suggestion.mapping_suggestion_id
        
        # Build OP text
        op_text = build_match_text(
            suggestion.op_league,
            suggestion.op_home,
            suggestion.op_away,
            # Use op_record categories if available
            op_record.category_tags if op_record else [],
        )
        
        if feedback.decision in (Decision.MATCH, Decision.SWAPPED):
            selected_id = feedback.selected_b365_match_id
            
            for cand in suggestion.candidates_top5:
                b365_text = self._build_b365_text(cand, b365_records)
                
                if cand.b365_match_id == selected_id:
                    # ── Positive pair ──
                    anchor = op_text
                    # If SWAPPED, use swapped OP text as anchor
                    if feedback.decision == Decision.SWAPPED:
                        anchor = build_swapped_text(
                            suggestion.op_league,
                            suggestion.op_home,
                            suggestion.op_away,
                            op_record.category_tags if op_record else [],
                        )
                    
                    pairs.append(TrainingPair(
                        anchor_text=anchor,
                        candidate_text=b365_text,
                        label=1.0,
                        is_hard_negative=False,
                        source_suggestion_id=sid,
                    ))
                else:
                    # ── Hard negative: was in Top-5 but NOT selected ──
                    # CRITICAL: skip if this is the true positive
                    # (prevent label contamination)
                    pairs.append(TrainingPair(
                        anchor_text=op_text,
                        candidate_text=b365_text,
                        label=0.0,
                        is_hard_negative=True,
                        source_suggestion_id=sid,
                    ))
        
        elif feedback.decision == Decision.NO_MATCH:
            # All candidates are hard negatives
            for cand in suggestion.candidates_top5:
                b365_text = self._build_b365_text(cand, b365_records)
                pairs.append(TrainingPair(
                    anchor_text=op_text,
                    candidate_text=b365_text,
                    label=0.0,
                    is_hard_negative=True,
                    source_suggestion_id=sid,
                ))
        
        return pairs
    
    def _build_b365_text(
        self,
        candidate,
        b365_records: Optional[Dict[str, MatchRecord]],
    ) -> str:
        """Build text for a B365 candidate."""
        if b365_records and candidate.b365_match_id in b365_records:
            rec = b365_records[candidate.b365_match_id]
            return build_match_text(
                rec.league, rec.home_team, rec.away_team, rec.category_tags
            )
        # Fallback: build from candidate fields
        return build_match_text(
            "", candidate.b365_home, candidate.b365_away, candidate.category_tags
        )
    
    # ── Getters ──
    
    def add_training_pair(self, pair: TrainingPair):
        """Add a training pair and update indexed sublists."""
        self._training_pairs.append(pair)
        if pair.label == 1.0:
            self._positives.append(pair)
        if pair.is_hard_negative:
            self._hard_negatives.append(pair)

    def get_training_pairs(self) -> List[TrainingPair]:
        """Return all training pairs."""
        return list(self._training_pairs)
    
    def get_positives(self) -> List[TrainingPair]:
        """Return only positive training pairs (O(1) via pre-indexed list)."""
        return list(self._positives)

    def get_hard_negatives(self) -> List[TrainingPair]:
        """Return only hard negative pairs (O(1) via pre-indexed list)."""
        return list(self._hard_negatives)
    
    def get_feedback_count(self) -> Dict[str, int]:
        """Count feedbacks by decision type."""
        counts = defaultdict(int)
        for fb in self._feedbacks.values():
            counts[fb.decision.value] += 1
        return dict(counts)
    
    def get_all_feedbacks(self) -> List[FeedbackRecord]:
        """Return all feedback records."""
        return list(self._feedbacks.values())
    
    def is_op_mapped(self, op_match_id: str) -> bool:
        """Check if OP match is already confirmed."""
        return op_match_id in self._op_to_b365
    
    def is_b365_mapped(self, b365_match_id: str) -> bool:
        """Check if B365 match is already confirmed."""
        return b365_match_id in self._b365_to_op


class BulkFeedbackLoader:
    """
    Load the existing ~10,000 human-labeled decisions from CSV/JSON/DB
    and convert them into training pairs.
    """
    
    @staticmethod
    def load_from_records(
        labeled_records: List[dict],
        store: FeedbackStore,
    ) -> int:
        """
        Load historical labeled records.
        
        Expected format per record:
        {
            "op_match_id": "...",
            "op_league": "...",
            "op_home": "...",
            "op_away": "...",
            "op_kickoff": "2024-01-01T15:00:00",
            "op_sport": "soccer",
            "b365_match_id": "..." or null,
            "b365_league": "...",
            "b365_home": "...",
            "b365_away": "...",
            "b365_kickoff": "2024-01-01T15:00:00",
            "decision": "MATCH" | "SWAPPED" | "NO_MATCH",
            "candidates": [  # optional: the Top-5 that were shown
                {"b365_match_id": "...", "b365_home": "...", "b365_away": "...", ...}
            ]
        }
        
        Returns number of training pairs created.
        """
        total_pairs = 0
        
        for i, rec in enumerate(labeled_records):
            try:
                decision = Decision(rec["decision"])
                
                # Build OP text
                op_cats = detect_cats_from_record(rec, prefix="op_")
                op_text = build_match_text(
                    rec.get("op_league", ""),
                    rec.get("op_home", ""),
                    rec.get("op_away", ""),
                    op_cats,
                )
                
                if decision in (Decision.MATCH, Decision.SWAPPED):
                    # Build positive pair
                    b365_cats = detect_cats_from_record(rec, prefix="b365_")
                    b365_text = build_match_text(
                        rec.get("b365_league", ""),
                        rec.get("b365_home", ""),
                        rec.get("b365_away", ""),
                        b365_cats,
                    )
                    
                    anchor = op_text
                    if decision == Decision.SWAPPED:
                        anchor = build_swapped_text(
                            rec.get("op_league", ""),
                            rec.get("op_home", ""),
                            rec.get("op_away", ""),
                            op_cats,
                        )
                    
                    # Positive pair
                    store.add_training_pair(TrainingPair(
                        anchor_text=anchor,
                        candidate_text=b365_text,
                        label=1.0,
                        is_hard_negative=False,
                    ))
                    total_pairs += 1
                    
                    # Hard negatives from unselected candidates
                    selected_id = rec.get("b365_match_id")
                    for cand in rec.get("candidates", []):
                        if cand.get("b365_match_id") == selected_id:
                            continue  # Skip true positive!
                        cand_cats = detect_cats_from_record(cand, prefix="b365_")
                        cand_text = build_match_text(
                            cand.get("b365_league", cand.get("league", "")),
                            cand.get("b365_home", cand.get("home_team", "")),
                            cand.get("b365_away", cand.get("away_team", "")),
                            cand_cats,
                        )
                        store.add_training_pair(TrainingPair(
                            anchor_text=op_text,
                            candidate_text=cand_text,
                            label=0.0,
                            is_hard_negative=True,
                        ))
                        total_pairs += 1
                
                elif decision == Decision.NO_MATCH:
                    # All candidates are hard negatives
                    for cand in rec.get("candidates", []):
                        cand_cats = detect_cats_from_record(cand, prefix="b365_")
                        cand_text = build_match_text(
                            cand.get("b365_league", cand.get("league", "")),
                            cand.get("b365_home", cand.get("home_team", "")),
                            cand.get("b365_away", cand.get("away_team", "")),
                            cand_cats,
                        )
                        store.add_training_pair(TrainingPair(
                            anchor_text=op_text,
                            candidate_text=cand_text,
                            label=0.0,
                            is_hard_negative=True,
                        ))
                        total_pairs += 1
            
            except Exception as e:
                logger.warning(f"Failed to process record {i}: {e}")
                continue
        
        logger.info(f"Bulk loaded {total_pairs} training pairs from {len(labeled_records)} records")
        return total_pairs


def detect_cats_from_record(rec: dict, prefix: str = "") -> List[str]:
    """Detect categories from a flat record dict."""
    from core.normalizer import detect_categories
    text_parts = [
        rec.get(f"{prefix}league", rec.get("league", "")),
        rec.get(f"{prefix}home", rec.get("home_team", "")),
        rec.get(f"{prefix}away", rec.get("away_team", "")),
    ]
    return detect_categories(" ".join(text_parts))


class CSEFeedbackLoader:
    """
    Fetches CSE team feedback from the remote API and converts it into
    training pairs for self-training the SBERT + Cross-Encoder models.

    Feedback mapping:
      Correct     → MATCH    → positive pair (boosts score)
      Not correct → NO_MATCH → hard negative (penalizes score)
      Need to swap→ SWAPPED  → positive pair with swapped OP text
      Not Sure    → skipped  (don't train on uncertain data)
    """

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=CONFIG.feedback_api.max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @classmethod
    def fetch_feedback(
        cls,
        platform: str = None,
        url: str = None,
    ) -> List[dict]:
        """
        Fetch all CSE feedback rows from the remote API.

        Args:
            platform: Platform filter (default: CONFIG.feedback_api.default_platform)
            url: Override API URL (default: from config prod/local toggle)

        Returns:
            List of raw feedback dicts from the API.
        """
        platform = platform or CONFIG.feedback_api.default_platform
        api_url = url or CONFIG.feedback_api.url

        session = cls._build_session()
        all_rows: List[dict] = []
        page = 1

        logger.info(f"Fetching CSE feedback from {api_url} (platform={platform})...")

        while True:
            params = {"platform": platform, "page": page, "limit": 100}
            try:
                resp = session.get(
                    api_url, params=params,
                    timeout=CONFIG.feedback_api.request_timeout,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to fetch feedback page {page}: {e}")
                break

            body = resp.json()

            if not body.get("status"):
                logger.error(f"Feedback API error: {body.get('message')}")
                break

            data = body.get("data", {})
            if isinstance(data, list):
                rows = data
                total_pages = 1
            else:
                rows = data.get("rows", data.get("data", []))
                total_pages = data.get("totalPages", 1)

            if not rows:
                break

            all_rows.extend(rows)
            logger.info(
                f"  Page {page}/{total_pages} — "
                f"fetched {len(rows)} feedback rows (total: {len(all_rows)})"
            )

            if page >= total_pages:
                break
            page += 1

        logger.info(f"Fetched {len(all_rows)} total CSE feedback rows")
        return all_rows

    @staticmethod
    def _extract_league_name(league_val) -> str:
        """Extract league name from nested object or plain string."""
        if isinstance(league_val, dict):
            return (
                league_val.get("name", "")
                or league_val.get("league_name_en", "")
            )
        return str(league_val) if league_val else ""

    @classmethod
    def _extract_provider_fields(cls, row: dict) -> dict:
        """
        Extract provider (OP/FlashScore/etc.) fields from a feedback row.
        Supports v2 format (nested provider_data array) and flat format.
        """
        provider_data = row.get("provider_data", [])
        p = provider_data[0] if provider_data and isinstance(provider_data, list) else {}

        home = (
            p.get("home_team", "")
            or row.get("home_team", row.get("op_home", ""))
        )
        away = (
            p.get("away_team", "")
            or row.get("away_team", row.get("op_away", ""))
        )
        league = (
            cls._extract_league_name(p.get("league"))
            or row.get("league", row.get("op_league", ""))
        )
        sport = (
            p.get("sport", "")
            or row.get("sport", row.get("op_sport", ""))
        )
        return {"home": home, "away": away, "league": league, "sport": sport}

    @classmethod
    def _extract_b365_fields(cls, row: dict) -> dict:
        """
        Extract Bet365 fields from a feedback row.
        Supports v2 format (nested bet365_match array) and flat format.
        """
        b365_data = row.get("bet365_match", [])
        b = b365_data[0] if b365_data and isinstance(b365_data, list) else {}

        home = (
            b.get("home_team", "")
            or row.get("bet365_home_team", row.get("b365_home", ""))
        )
        away = (
            b.get("away_team", "")
            or row.get("bet365_away_team", row.get("b365_away", ""))
        )
        league = (
            cls._extract_league_name(b.get("league"))
            or row.get("bet365_league", row.get("b365_league", ""))
        )
        return {"home": home, "away": away, "league": league}

    @classmethod
    def convert_to_training_pairs(
        cls,
        feedback_rows: List[dict],
        store: "FeedbackStore",
    ) -> Tuple[int, Dict[str, int]]:
        """
        Convert CSE feedback rows into training pairs.

        Supports both v2 format (nested provider_data/bet365_match arrays)
        and legacy flat format (home_team, bet365_home_team at root level).

        For each feedback row:
          - Correct:     Build (OP text, B365 text) positive pair (label=1.0)
          - Not correct: Build (OP text, B365 text) hard negative (label=0.0)
          - Need to swap: Build (OP swapped text, B365 text) positive pair (label=1.0)
          - Not Sure:    Skip — no training signal from uncertain feedback

        Args:
            feedback_rows: Raw dicts from the CSE feedback API
            store: FeedbackStore to append training pairs to

        Returns:
            (total_pairs_created, counts_by_feedback_type)
        """
        from core.normalizer import detect_categories

        total_pairs = 0
        counts: Dict[str, int] = {
            "correct": 0,
            "not_correct": 0,
            "need_to_swap": 0,
            "not_sure_skipped": 0,
            "no_b365_skipped": 0,
            "errors": 0,
        }

        for i, row in enumerate(feedback_rows):
            try:
                feedback_val = row.get("feedback", row.get("cse_feedback", ""))
                if not feedback_val:
                    counts["errors"] += 1
                    continue

                try:
                    cse_feedback = CSEFeedback(feedback_val)
                except ValueError:
                    feedback_lower = feedback_val.strip().lower()
                    LOOSE_MAP = {
                        "correct": CSEFeedback.CORRECT,
                        "not sure": CSEFeedback.NOT_SURE,
                        "not correct": CSEFeedback.NOT_CORRECT,
                        "need to swap": CSEFeedback.NEED_TO_SWAP,
                        "swap": CSEFeedback.NEED_TO_SWAP,
                        "wrong": CSEFeedback.NOT_CORRECT,
                    }
                    cse_feedback = LOOSE_MAP.get(feedback_lower)
                    if cse_feedback is None:
                        logger.warning(
                            f"Row {i}: Unknown feedback value '{feedback_val}', skipping"
                        )
                        counts["errors"] += 1
                        continue

                decision = CSEFeedback.to_decision(cse_feedback)

                if decision is None:
                    counts["not_sure_skipped"] += 1
                    continue

                prov = cls._extract_provider_fields(row)
                op_home = prov["home"]
                op_away = prov["away"]
                op_league = prov["league"]

                b365 = cls._extract_b365_fields(row)
                b365_home = b365["home"]
                b365_away = b365["away"]
                b365_league = b365["league"]

                if not (op_home and op_away):
                    logger.warning(f"Row {i}: Missing provider team names, skipping")
                    counts["errors"] += 1
                    continue

                if not (b365_home and b365_away):
                    counts["no_b365_skipped"] += 1
                    continue

                op_cats = detect_categories(f"{op_league} {op_home} {op_away}")
                b365_cats = detect_categories(f"{b365_league} {b365_home} {b365_away}")

                op_text = build_match_text(op_league, op_home, op_away, op_cats)
                b365_text = build_match_text(b365_league, b365_home, b365_away, b365_cats)

                provider_id = str(row.get("provider_id", row.get("op_match_id", f"cse_{i}")))

                if decision == Decision.MATCH:
                    store.add_training_pair(TrainingPair(
                        anchor_text=op_text,
                        candidate_text=b365_text,
                        label=1.0,
                        is_hard_negative=False,
                        source_suggestion_id=f"cse_feedback_{provider_id}",
                    ))
                    total_pairs += 1
                    counts["correct"] += 1

                elif decision == Decision.NO_MATCH:
                    store.add_training_pair(TrainingPair(
                        anchor_text=op_text,
                        candidate_text=b365_text,
                        label=0.0,
                        is_hard_negative=True,
                        source_suggestion_id=f"cse_feedback_{provider_id}",
                    ))
                    total_pairs += 1
                    counts["not_correct"] += 1

                elif decision == Decision.SWAPPED:
                    swapped_text = build_swapped_text(
                        op_league, op_home, op_away, op_cats
                    )
                    store.add_training_pair(TrainingPair(
                        anchor_text=swapped_text,
                        candidate_text=b365_text,
                        label=1.0,
                        is_hard_negative=False,
                        source_suggestion_id=f"cse_feedback_{provider_id}",
                    ))
                    store.add_training_pair(TrainingPair(
                        anchor_text=op_text,
                        candidate_text=b365_text,
                        label=0.0,
                        is_hard_negative=True,
                        source_suggestion_id=f"cse_feedback_{provider_id}",
                    ))
                    total_pairs += 2
                    counts["need_to_swap"] += 1

            except Exception as e:
                logger.warning(f"Failed to process CSE feedback row {i}: {e}")
                counts["errors"] += 1
                continue

        logger.info(
            f"CSE feedback → {total_pairs} training pairs "
            f"(correct={counts['correct']}, not_correct={counts['not_correct']}, "
            f"swapped={counts['need_to_swap']}, "
            f"skipped_not_sure={counts['not_sure_skipped']}, "
            f"skipped_no_b365={counts['no_b365_skipped']}, "
            f"errors={counts['errors']})"
        )
        return total_pairs, counts

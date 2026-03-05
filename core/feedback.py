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
    
    def deduplicate_conflicting_pairs(self) -> int:
        """
        Resolve label contamination where the same (anchor, candidate) text
        appears as BOTH a positive and a negative pair.

        This happens when different reviewers give conflicting feedback for
        the same match pair, or when a match is first marked 'Not correct'
        and later 'Mapping Completed' (or vice versa).

        Resolution strategy: positive (Correct/Mapping Completed) wins over
        negative, because 'Mapping Completed' is a confirmed final state.
        The negative duplicate is removed.

        Returns:
            Number of contaminated pairs removed.
        """
        # Build set of positive pair keys
        positive_keys = set()
        for p in self._positives:
            positive_keys.add((p.anchor_text, p.candidate_text))

        if not positive_keys:
            return 0

        # Find negatives that conflict with positives
        contaminated = []
        for i, p in enumerate(self._training_pairs):
            if p.label == 0.0 and (p.anchor_text, p.candidate_text) in positive_keys:
                contaminated.append(i)

        if not contaminated:
            return 0

        # Remove contaminated pairs (iterate in reverse to preserve indices)
        for i in reversed(contaminated):
            self._training_pairs.pop(i)

        # Rebuild hard_negatives index
        self._hard_negatives = [
            p for p in self._training_pairs if p.is_hard_negative
        ]

        logger.info(
            f"Deduplication: removed {len(contaminated)} conflicting negative pairs "
            f"that also appeared as positives (positive label wins)"
        )
        return len(contaminated)

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
    def _fetch_page(
        cls,
        api_url: str,
        platform: str,
        page: int,
        session: requests.Session = None,
    ) -> Tuple[int, List[dict], Optional[int]]:
        """
        Fetch a single page of feedback from the API.

        Returns:
            (page_number, rows, total_pages)
            total_pages is None if it couldn't be determined.
        """
        _session = session or cls._build_session()
        params = {
            "platform": platform,
            "page": page,
            "limit": CONFIG.feedback_api.page_size,
        }
        try:
            resp = _session.get(
                api_url, params=params,
                timeout=CONFIG.feedback_api.request_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch feedback page {page}: {e}")
            return page, [], None

        body = resp.json()

        if not body.get("status"):
            logger.error(f"Feedback API error on page {page}: {body.get('message')}")
            return page, [], None

        data = body.get("data", {})
        if isinstance(data, list):
            return page, data, 1
        else:
            rows = data.get("rows", data.get("data", []))
            total_pages = data.get("totalPages", 1)
            return page, rows, total_pages

    @classmethod
    def fetch_feedback(
        cls,
        platform: str = None,
        url: str = None,
    ) -> List[dict]:
        """
        Fetch all CSE feedback rows from the remote API using parallel requests.

        Phase 1: Fetch page 1 sequentially to discover totalPages.
        Phase 2: Fetch remaining pages in parallel using ThreadPoolExecutor.

        Args:
            platform: Platform filter (default: CONFIG.feedback_api.default_platform)
            url: Override API URL (default: from config prod/local toggle)

        Returns:
            List of raw feedback dicts from the API.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        platform = platform or CONFIG.feedback_api.default_platform
        api_url = url or CONFIG.feedback_api.url
        max_workers = CONFIG.feedback_api.max_parallel_workers

        logger.info(f"Fetching CSE feedback from {api_url} (platform={platform})...")
        t_start = _time.monotonic()

        # ── Phase 1: Fetch page 1 to discover totalPages ──
        session = cls._build_session()
        page_num, first_rows, total_pages = cls._fetch_page(
            api_url, platform, page=1, session=session,
        )
        if not first_rows:
            logger.warning("Page 1 returned no rows. Nothing to fetch.")
            return []

        total_pages = total_pages or 1
        total_pages = min(total_pages, CONFIG.feedback_api.max_pages)

        logger.info(
            f"  Page 1/{total_pages} — fetched {len(first_rows)} rows. "
            f"Fetching remaining {total_pages - 1} pages with {max_workers} workers..."
        )

        # ── Phase 2: Fetch remaining pages in parallel ──
        # Use dict keyed by page number to maintain order
        results_by_page: Dict[int, List[dict]] = {1: first_rows}

        if total_pages > 1:
            remaining_pages = list(range(2, total_pages + 1))

            def _worker(pg: int) -> Tuple[int, List[dict]]:
                """Fetch one page with its own session for thread safety."""
                worker_session = cls._build_session()
                _, rows, _ = cls._fetch_page(
                    api_url, platform, page=pg, session=worker_session,
                )
                return pg, rows

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_worker, pg): pg
                    for pg in remaining_pages
                }

                completed = 0
                failed = 0
                for future in as_completed(futures):
                    pg = futures[future]
                    try:
                        page_num, rows = future.result()
                        if rows:
                            results_by_page[page_num] = rows
                        else:
                            failed += 1
                        completed += 1
                        # Log progress every 20 pages
                        if completed % 20 == 0 or completed == len(remaining_pages):
                            fetched_so_far = sum(len(r) for r in results_by_page.values())
                            logger.info(
                                f"  Progress: {completed}/{len(remaining_pages)} pages "
                                f"({fetched_so_far} rows, {failed} failed)"
                            )
                    except Exception as e:
                        failed += 1
                        completed += 1
                        logger.error(f"  Page {pg} raised exception: {e}")

                if failed > 0:
                    logger.warning(f"  {failed}/{len(remaining_pages)} pages failed to fetch")

        # ── Combine results in page order ──
        all_rows: List[dict] = []
        for pg in sorted(results_by_page.keys()):
            all_rows.extend(results_by_page[pg])

        elapsed = _time.monotonic() - t_start
        logger.info(
            f"Fetched {len(all_rows)} total CSE feedback rows "
            f"from {len(results_by_page)} pages in {elapsed:.1f}s "
            f"(was ~{total_pages * 5.5:.0f}s sequential)"
        )
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

    @staticmethod
    def _extract_feedback_from_logs(row: dict) -> Optional[str]:
        """
        Extract the effective reviewer decision from logs.

        Log entries like "Mapping Completed" or "Team Switched" are
        status transitions, not review decisions.  When the latest entry
        is a status transition we walk backwards through the sorted logs
        to find the most recent *review decision* and then combine it
        with the status to determine the correct training signal.

        Decision hierarchy (latest timestamp wins):
          - "Mapping Completed" → the mapping was confirmed correct
            UNLESS an earlier decision was "Not correct" and there is
            no subsequent corrective action.  In practice, if a mapping
            is completed it means a reviewer confirmed it.
          - "Not correct", "Not Sure", "Need to swap", "Correct" → direct decisions
        """
        logs = row.get("logs")
        if not isinstance(logs, list) or not logs:
            return None

        valid_logs = [entry for entry in logs if isinstance(entry, dict)]
        if not valid_logs:
            return None

        def _log_sort_key(entry: dict) -> str:
            return str(entry.get("when") or "")

        sorted_logs = sorted(valid_logs, key=_log_sort_key)
        latest_entry = sorted_logs[-1]
        latest_what = str(latest_entry.get("what", "")).strip()

        # Status transitions that indicate a final confirmed state
        STATUS_TRANSITIONS = {
            "Mapping Completed",
            "mapping completed",
            "Mapped",
            "mapped",
        }

        SWAP_TRANSITIONS = {
            "Team Switched",
            "team switched",
            "Switch Completed",
            "switch completed",
            "Teams Swapped",
        }

        if latest_what in STATUS_TRANSITIONS:
            # "Mapping Completed" means the match was confirmed as correct.
            # Check is_team_switched_completed to see if it was a swap.
            if row.get("is_team_switched_completed"):
                return "Need to swap"
            return "Correct"

        if latest_what in SWAP_TRANSITIONS:
            return "Need to swap"

        return latest_what if latest_what else None

    @classmethod
    def _extract_feedback_value(cls, row: dict) -> str:
        """
        Training signal source priority:
        1) latest logs[].what from CSE review history (with status transition handling)
        2) row-level status fields (is_mapped_completed, is_team_switched_completed)
        3) root feedback/cse_feedback fallback for backward compatibility
        """
        from_logs = cls._extract_feedback_from_logs(row)
        if from_logs:
            return from_logs

        # Fallback: use structured status fields
        if row.get("is_mapped_completed"):
            if row.get("is_team_switched_completed"):
                return "Need to swap"
            return "Correct"

        fallback = row.get("feedback", row.get("cse_feedback", ""))
        return str(fallback).strip() if fallback else ""

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
            "no_feedback_value": 0,
            "unknown_feedback_value": 0,
            "errors": 0,
        }
        # Track unique unknown feedback values for diagnostics
        unknown_values: Dict[str, int] = {}

        for i, row in enumerate(feedback_rows):
            try:
                feedback_val = cls._extract_feedback_value(row)
                if not feedback_val:
                    counts["no_feedback_value"] += 1
                    continue

                try:
                    cse_feedback = CSEFeedback(feedback_val)
                except ValueError:
                    feedback_lower = feedback_val.strip().lower()
                    LOOSE_MAP = {
                        "correct": CSEFeedback.CORRECT,
                        "not sure": CSEFeedback.NOT_SURE,
                        "not_sure": CSEFeedback.NOT_SURE,
                        "not correct": CSEFeedback.NOT_CORRECT,
                        "notcorrect": CSEFeedback.NOT_CORRECT,
                        "not_correct": CSEFeedback.NOT_CORRECT,
                        "need to swap": CSEFeedback.NEED_TO_SWAP,
                        "need_to_swap": CSEFeedback.NEED_TO_SWAP,
                        "swap": CSEFeedback.NEED_TO_SWAP,
                        "wrong": CSEFeedback.NOT_CORRECT,
                        # Status transitions from CSE review workflow
                        "mapping completed": CSEFeedback.CORRECT,
                        "mapped": CSEFeedback.CORRECT,
                        "team switched": CSEFeedback.NEED_TO_SWAP,
                        "switch completed": CSEFeedback.NEED_TO_SWAP,
                        "teams swapped": CSEFeedback.NEED_TO_SWAP,
                    }
                    cse_feedback = LOOSE_MAP.get(feedback_lower)
                    if cse_feedback is None:
                        unknown_values[feedback_val] = unknown_values.get(feedback_val, 0) + 1
                        counts["unknown_feedback_value"] += 1
                        if len(unknown_values) <= 20:
                            logger.warning(
                                f"Row {i}: Unknown feedback value '{feedback_val}', skipping"
                            )
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

        # ── Deduplicate conflicting pairs (same pair as both pos + neg) ──
        removed = store.deduplicate_conflicting_pairs()
        if removed > 0:
            total_pairs -= removed
            counts["contamination_resolved"] = removed

        # Log unknown feedback values for diagnostics
        if unknown_values:
            logger.warning(
                f"Unknown feedback values encountered ({len(unknown_values)} unique): "
                f"{dict(sorted(unknown_values.items(), key=lambda x: -x[1])[:10])}"
            )

        # Diagnostic: alert if zero positives
        if counts["correct"] == 0 and counts["need_to_swap"] == 0:
            total_rows = len(feedback_rows)
            accounted = (
                counts["correct"] + counts["not_correct"] + counts["need_to_swap"]
                + counts["not_sure_skipped"] + counts["no_b365_skipped"]
                + counts["no_feedback_value"] + counts["unknown_feedback_value"]
                + counts["errors"]
            )
            unaccounted = total_rows - accounted
            logger.error(
                f"ZERO POSITIVES: {total_rows} feedback rows fetched but 0 marked 'Correct'. "
                f"Breakdown: not_correct={counts['not_correct']}, "
                f"not_sure={counts['not_sure_skipped']}, "
                f"no_b365={counts['no_b365_skipped']}, "
                f"no_feedback_value={counts['no_feedback_value']}, "
                f"unknown_values={counts['unknown_feedback_value']}, "
                f"errors={counts['errors']}, "
                f"unaccounted={unaccounted}. "
                f"Check if the CSE team is marking matches as 'Correct' or if "
                f"the feedback API response format has changed."
            )

        logger.info(
            f"CSE feedback → {total_pairs} training pairs "
            f"(correct={counts['correct']}, not_correct={counts['not_correct']}, "
            f"swapped={counts['need_to_swap']}, "
            f"skipped_not_sure={counts['not_sure_skipped']}, "
            f"skipped_no_b365={counts['no_b365_skipped']}, "
            f"no_feedback_value={counts['no_feedback_value']}, "
            f"unknown_feedback_value={counts['unknown_feedback_value']}, "
            f"errors={counts['errors']})"
        )
        return total_pairs, counts

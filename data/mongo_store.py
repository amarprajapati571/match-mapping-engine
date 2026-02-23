"""
MongoDB-backed storage for production.
Drop-in replacement for in-memory FeedbackStore.

Collections:
- suggestions: cached MappingSuggestion documents
- feedbacks: human decisions with 1:1 constraint indexes
- training_pairs: generated training data
- eval_results: evaluation history for drift monitoring
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from config.settings import CONFIG
from core.models import (
    Decision, FeedbackRecord, MappingSuggestion,
    TrainingPair, EvalResult,
)

logger = logging.getLogger(__name__)


class MongoStore:
    """
    Production-grade MongoDB storage with:
    - 1:1 mapping constraint via unique indexes
    - Idempotent feedback ingestion
    - Training pair generation
    - Eval result history
    """
    
    def __init__(self, uri: str = None, db_name: str = None):
        self.client = MongoClient(uri or CONFIG.mongo_uri)
        self.db = self.client[db_name or CONFIG.mongo_db]
        
        # Collections
        self.suggestions = self.db["suggestions"]
        self.feedbacks = self.db["feedbacks"]
        self.training_pairs = self.db["training_pairs"]
        self.eval_results = self.db["eval_results"]
        self.mappings = self.db["confirmed_mappings"]
        
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create indexes for constraints and query performance."""
        # Suggestions: lookup by ID
        self.suggestions.create_index("mapping_suggestion_id", unique=True)
        self.suggestions.create_index("op_match_id")
        
        # Feedbacks: one decision per suggestion
        self.feedbacks.create_index("mapping_suggestion_id", unique=True)
        self.feedbacks.create_index("op_match_id")
        self.feedbacks.create_index("decision")
        
        # 1:1 mapping constraint
        self.mappings.create_index("op_match_id", unique=True)
        self.mappings.create_index("b365_match_id", unique=True)
        
        # Training pairs
        self.training_pairs.create_index("source_suggestion_id")
        self.training_pairs.create_index("label")
        self.training_pairs.create_index("created_at")
        
        # Eval results: history
        self.eval_results.create_index("evaluated_at", unique=False)
        self.eval_results.create_index("model_version")
        
        logger.info("MongoDB indexes ensured.")
    
    # ── Suggestions ──
    
    def store_suggestion(self, suggestion: MappingSuggestion):
        """Store a mapping suggestion (upsert)."""
        doc = suggestion.model_dump()
        doc["_id"] = suggestion.mapping_suggestion_id
        self.suggestions.replace_one(
            {"_id": doc["_id"]}, doc, upsert=True
        )
    
    def get_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """Retrieve a suggestion by ID."""
        return self.suggestions.find_one(
            {"mapping_suggestion_id": suggestion_id},
            {"_id": 0},
        )
    
    # ── Feedback + 1:1 Constraint ──
    
    def ingest_feedback(
        self, feedback: FeedbackRecord
    ) -> Tuple[bool, str]:
        """
        Ingest feedback with 1:1 mapping enforcement.
        Uses MongoDB transactions for atomicity.
        """
        sid = feedback.mapping_suggestion_id
        
        # Verify suggestion exists
        suggestion = self.get_suggestion(sid)
        if not suggestion:
            return False, f"Unknown suggestion_id: {sid}"
        
        # Check for duplicate feedback
        existing = self.feedbacks.find_one({"mapping_suggestion_id": sid})
        if existing:
            return False, f"Feedback already exists for suggestion {sid}"
        
        if feedback.decision in (Decision.MATCH, Decision.SWAPPED):
            b365_id = feedback.selected_b365_match_id
            if not b365_id:
                return False, "MATCH/SWAPPED requires selected_b365_match_id"
            
            # Enforce 1:1 via unique index
            try:
                self.mappings.insert_one({
                    "op_match_id": feedback.op_match_id,
                    "b365_match_id": b365_id,
                    "decision": feedback.decision.value,
                    "suggestion_id": sid,
                    "created_at": datetime.utcnow(),
                })
            except DuplicateKeyError as e:
                return False, f"1:1 constraint violated: {e}"
        
        # Store feedback
        doc = feedback.model_dump()
        doc["_id"] = sid
        self.feedbacks.insert_one(doc)
        
        # Generate and store training pairs
        pairs = self._generate_training_pairs(feedback, suggestion)
        if pairs:
            self.training_pairs.insert_many(
                [p.model_dump() for p in pairs]
            )
        
        logger.info(
            f"Feedback stored: {feedback.decision.value} → "
            f"{len(pairs)} training pairs"
        )
        return True, f"OK: {len(pairs)} training pairs"
    
    def _generate_training_pairs(
        self, feedback: FeedbackRecord, suggestion: dict
    ) -> List[TrainingPair]:
        """Generate training pairs from feedback + cached suggestion."""
        from core.normalizer import build_match_text, build_swapped_text
        
        pairs = []
        sid = suggestion["mapping_suggestion_id"]
        
        op_text = build_match_text(
            suggestion.get("op_league", ""),
            suggestion.get("op_home", ""),
            suggestion.get("op_away", ""),
        )
        
        candidates = suggestion.get("candidates_top5", [])
        
        if feedback.decision in (Decision.MATCH, Decision.SWAPPED):
            selected_id = feedback.selected_b365_match_id
            
            for cand in candidates:
                cand_text = build_match_text(
                    "", cand.get("b365_home", ""), cand.get("b365_away", ""),
                    cand.get("category_tags", []),
                )
                
                if cand.get("b365_match_id") == selected_id:
                    anchor = op_text
                    if feedback.decision == Decision.SWAPPED:
                        anchor = build_swapped_text(
                            suggestion.get("op_league", ""),
                            suggestion.get("op_home", ""),
                            suggestion.get("op_away", ""),
                        )
                    pairs.append(TrainingPair(
                        anchor_text=anchor,
                        candidate_text=cand_text,
                        label=1.0,
                        source_suggestion_id=sid,
                    ))
                else:
                    pairs.append(TrainingPair(
                        anchor_text=op_text,
                        candidate_text=cand_text,
                        label=0.0,
                        is_hard_negative=True,
                        source_suggestion_id=sid,
                    ))
        
        elif feedback.decision == Decision.NO_MATCH:
            for cand in candidates:
                cand_text = build_match_text(
                    "", cand.get("b365_home", ""), cand.get("b365_away", ""),
                    cand.get("category_tags", []),
                )
                pairs.append(TrainingPair(
                    anchor_text=op_text,
                    candidate_text=cand_text,
                    label=0.0,
                    is_hard_negative=True,
                    source_suggestion_id=sid,
                ))
        
        return pairs
    
    # ── Training Data Export ──
    
    def get_training_pairs(
        self, limit: int = 0, label: Optional[float] = None
    ) -> List[dict]:
        """Export training pairs."""
        query = {}
        if label is not None:
            query["label"] = label
        
        cursor = self.training_pairs.find(query, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def get_training_pair_count(self) -> Dict[str, int]:
        """Count training pairs by type."""
        pipeline = [
            {"$group": {
                "_id": "$label",
                "count": {"$sum": 1}
            }}
        ]
        result = list(self.training_pairs.aggregate(pipeline))
        return {
            "positive": next((r["count"] for r in result if r["_id"] == 1.0), 0),
            "negative": next((r["count"] for r in result if r["_id"] == 0.0), 0),
            "total": sum(r["count"] for r in result),
        }
    
    # ── Evaluation History ──
    
    def store_eval_result(self, result: EvalResult):
        """Store an evaluation result."""
        doc = result.model_dump()
        doc["_id"] = result.eval_id
        self.eval_results.insert_one(doc)
    
    def get_eval_history(self, limit: int = 10) -> List[dict]:
        """Get recent evaluation results."""
        return list(
            self.eval_results
            .find({}, {"_id": 0})
            .sort("evaluated_at", DESCENDING)
            .limit(limit)
        )
    
    def get_feedback_count(self) -> Dict[str, int]:
        """Count feedbacks by decision type."""
        pipeline = [
            {"$group": {"_id": "$decision", "count": {"$sum": 1}}}
        ]
        result = list(self.feedbacks.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in result}

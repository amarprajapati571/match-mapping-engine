"""
FastAPI Application — Production Inference API.

Endpoints:
1. POST /predict         → Top-5 candidates + gate decision
2. POST /predict/batch   → Batch predictions
3. POST /feedback        → Ingest human decisions
4. POST /index/refresh   → Re-index B365 pool
5. GET  /health          → Health check
6. GET  /metrics         → Current feedback stats
7. POST /evaluate        → Run offline evaluation
8. POST /models/reload   → Hot-reload models (feature flag)
"""

import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config.settings import CONFIG
from core.models import (
    MatchRecord, MappingSuggestion, FeedbackRecord,
    Decision, GateResult, EvalResult,
)
from core.inference import InferenceEngine
from core.feedback import FeedbackStore, CSEFeedbackLoader
from training.trainer import TrainingOrchestrator, DatasetBuilder

logger = logging.getLogger(__name__)

# ── Global State ──
engine: Optional[InferenceEngine] = None
feedback_store: Optional[FeedbackStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine and stores on startup."""
    global engine, feedback_store
    
    logger.info("Initializing Match Mapping Engine...")
    engine = InferenceEngine()
    feedback_store = FeedbackStore()
    logger.info("Engine ready. Awaiting B365 index (call POST /index/refresh).")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Match Mapping Engine",
    description=(
        "SBERT retrieval + Cross-Encoder reranking for sports match mapping. "
        "Returns Top-5 candidates with auto-match gate decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════
# Request/Response Schemas
# ═══════════════════════════════════════════════

class PredictRequest(BaseModel):
    op_match: MatchRecord


class BatchPredictRequest(BaseModel):
    op_matches: List[MatchRecord]


class IndexRefreshRequest(BaseModel):
    """B365 matches to index."""
    b365_matches: List[MatchRecord]


class FeedbackRequest(BaseModel):
    mapping_suggestion_id: str
    op_match_id: str
    decision: Decision
    selected_b365_match_id: Optional[str] = None
    swapped: bool = False
    reason_code: Optional[str] = None
    reviewer_id: Optional[str] = None


class ModelReloadRequest(BaseModel):
    use_tuned_sbert: Optional[bool] = None
    use_tuned_cross_encoder: Optional[bool] = None
    tuned_sbert_path: Optional[str] = None
    tuned_cross_encoder_path: Optional[str] = None


class SelfTrainRequest(BaseModel):
    platform: str = "ODDSPORTAL"
    use_local_api: bool = False
    api_url: Optional[str] = None
    sbert_only: bool = False
    ce_only: bool = False
    dry_run: bool = False
    auto_reload: bool = True


class HealthResponse(BaseModel):
    status: str
    b365_pool_size: int
    feedback_count: dict
    models: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════

@app.post("/predict", response_model=MappingSuggestion)
async def predict(request: PredictRequest):
    """
    Generate Top-5 mapping candidates for an OP match.
    Returns candidates with scores + AUTO_MATCH / NEED_REVIEW decision.
    """
    if engine._b365_index is None:
        raise HTTPException(
            status_code=503,
            detail="B365 index not built. Call POST /index/refresh first."
        )
    
    suggestion = engine.predict(request.op_match)
    
    # Cache suggestion for later feedback processing
    feedback_store.store_suggestion(suggestion)
    
    return suggestion


@app.post("/predict/batch", response_model=List[MappingSuggestion])
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple OP matches."""
    if engine._b365_index is None:
        raise HTTPException(status_code=503, detail="B365 index not built.")
    
    suggestions = engine.predict_batch(request.op_matches)
    
    for s in suggestions:
        feedback_store.store_suggestion(s)
    
    return suggestions


@app.post("/index/refresh")
async def refresh_index(request: IndexRefreshRequest):
    """Re-index the B365 match pool."""
    engine.index_b365_pool(request.b365_matches)
    return {
        "status": "ok",
        "pool_size": len(request.b365_matches),
        "message": f"Indexed {len(request.b365_matches)} B365 matches."
    }


@app.post("/feedback")
async def ingest_feedback(request: FeedbackRequest):
    """
    Ingest a human decision from the Admin UI.
    Automatically generates training pairs (positive + hard negatives).
    """
    feedback = FeedbackRecord(
        mapping_suggestion_id=request.mapping_suggestion_id,
        op_match_id=request.op_match_id,
        decision=request.decision,
        selected_b365_match_id=request.selected_b365_match_id,
        swapped=request.swapped,
        reason_code=request.reason_code,
        reviewer_id=request.reviewer_id,
    )
    
    success, message = feedback_store.ingest_feedback(feedback)
    
    if not success:
        raise HTTPException(status_code=409, detail=message)
    
    return {"status": "ok", "message": message}


@app.post("/models/reload")
async def reload_models(request: ModelReloadRequest):
    """
    Hot-reload models with feature flag support.
    Toggle between base and tuned models without restart.
    """
    if request.use_tuned_sbert is not None:
        CONFIG.model.use_tuned_sbert = request.use_tuned_sbert
    if request.use_tuned_cross_encoder is not None:
        CONFIG.model.use_tuned_cross_encoder = request.use_tuned_cross_encoder
    if request.tuned_sbert_path:
        CONFIG.model.tuned_sbert_path = request.tuned_sbert_path
    if request.tuned_cross_encoder_path:
        CONFIG.model.tuned_cross_encoder_path = request.tuned_cross_encoder_path
    
    engine.reload_models()
    
    return {
        "status": "ok",
        "active_sbert": (
            CONFIG.model.tuned_sbert_path
            if CONFIG.model.use_tuned_sbert
            else CONFIG.model.sbert_model
        ),
        "active_cross_encoder": (
            CONFIG.model.tuned_cross_encoder_path
            if CONFIG.model.use_tuned_cross_encoder
            else CONFIG.model.cross_encoder_model
        ),
    }


@app.post("/self-train")
async def self_train(request: SelfTrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger self-training from CSE team feedback.

    Fetches feedback from the CSE API, converts to training pairs,
    trains SBERT + Cross-Encoder, and optionally hot-reloads models.

    Feedback mapping:
      Correct     → positive pair (boosts score: e.g. 9.0 → 9.5)
      Not correct → hard negative (lowers score: e.g. 8.5 → 7.0)
      Need to swap → positive with swapped teams
      Not Sure    → skipped
    """
    if request.use_local_api:
        CONFIG.feedback_api.use_local = True

    feedback_rows = CSEFeedbackLoader.fetch_feedback(
        platform=request.platform,
        url=request.api_url,
    )

    if not feedback_rows:
        return {
            "status": "no_data",
            "message": "No feedback rows returned from CSE API.",
        }

    store = FeedbackStore()
    n_pairs, counts = CSEFeedbackLoader.convert_to_training_pairs(
        feedback_rows, store
    )

    trainable = counts["correct"] + counts["not_correct"] + counts["need_to_swap"]

    if trainable < CONFIG.feedback_api.min_feedback_for_training:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {trainable} trainable feedback rows "
                f"(need {CONFIG.feedback_api.min_feedback_for_training}). "
                f"Collect more CSE reviews."
            ),
            "feedback_counts": counts,
        }

    if request.dry_run:
        return {
            "status": "dry_run",
            "feedback_rows": len(feedback_rows),
            "training_pairs": n_pairs,
            "feedback_counts": counts,
            "positives": len(store.get_positives()),
            "hard_negatives": len(store.get_hard_negatives()),
        }

    all_pairs = store.get_training_pairs()
    if not DatasetBuilder.validate_no_contamination(all_pairs):
        return {
            "status": "error",
            "message": "Label contamination detected in training data.",
        }

    def _run_training():
        import os
        from datetime import datetime as dt

        ts = dt.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)

        sbert_out = f"models/sbert_cse_tuned_{ts}"
        ce_out = f"models/ce_cse_tuned_{ts}"

        orchestrator = TrainingOrchestrator(all_pairs)
        orchestrator.prepare()

        if request.ce_only:
            orchestrator.train_cross_encoder(ce_out)
        elif request.sbert_only:
            orchestrator.train_sbert(sbert_out)
        else:
            orchestrator.train_all(sbert_out, ce_out)

        if request.auto_reload and engine:
            if not request.ce_only:
                CONFIG.model.tuned_sbert_path = sbert_out
                CONFIG.model.use_tuned_sbert = True
            if not request.sbert_only:
                CONFIG.model.tuned_cross_encoder_path = ce_out
                CONFIG.model.use_tuned_cross_encoder = True
            engine.reload_models()
            logger.info("Models auto-reloaded after self-training.")

    background_tasks.add_task(_run_training)

    return {
        "status": "training_started",
        "message": "Self-training started in background.",
        "feedback_rows": len(feedback_rows),
        "training_pairs": n_pairs,
        "feedback_counts": counts,
        "auto_reload": request.auto_reload,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with system status."""
    return HealthResponse(
        status="healthy",
        b365_pool_size=len(engine._b365_records) if engine else 0,
        feedback_count=feedback_store.get_feedback_count() if feedback_store else {},
        models={
            "sbert": (
                CONFIG.model.tuned_sbert_path
                if CONFIG.model.use_tuned_sbert
                else CONFIG.model.sbert_model
            ),
            "cross_encoder": (
                CONFIG.model.tuned_cross_encoder_path
                if CONFIG.model.use_tuned_cross_encoder
                else CONFIG.model.cross_encoder_model
            ),
            "use_tuned_sbert": CONFIG.model.use_tuned_sbert,
            "use_tuned_cross_encoder": CONFIG.model.use_tuned_cross_encoder,
        },
    )


@app.get("/metrics")
async def metrics():
    """Get current system metrics."""
    return {
        "feedback_counts": feedback_store.get_feedback_count(),
        "training_pairs_total": len(feedback_store.get_training_pairs()),
        "training_pairs_positive": len(feedback_store.get_positives()),
        "training_pairs_hard_negative": len(feedback_store.get_hard_negatives()),
        "b365_pool_size": len(engine._b365_records),
        "active_models": {
            "sbert": CONFIG.model.sbert_model,
            "cross_encoder": CONFIG.model.cross_encoder_model,
            "tuned_sbert": CONFIG.model.use_tuned_sbert,
            "tuned_ce": CONFIG.model.use_tuned_cross_encoder,
        },
    }


@app.get("/training-pairs")
async def get_training_pairs(limit: int = 100, offset: int = 0):
    """Export training pairs for inspection."""
    pairs = feedback_store.get_training_pairs()
    return {
        "total": len(pairs),
        "pairs": [p.model_dump() for p in pairs[offset:offset + limit]],
    }

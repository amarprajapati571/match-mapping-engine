"""
Configuration settings for the AI Match Mapping Engine.
All thresholds and model paths are centralized here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)

_logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        _logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _logger.info("Apple MPS detected")
        return "mps"
    _logger.info("No GPU detected — using CPU")
    return "cpu"


DEVICE = os.getenv("DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = _detect_device()


@dataclass
class ModelConfig:
    """Model paths and parameters."""
    # ── Free Models (HuggingFace) ──
    sbert_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Fine-tuned model paths (after training) ──
    tuned_sbert_path: Optional[str] = None
    tuned_cross_encoder_path: Optional[str] = None

    # Feature flag: use tuned models in production
    use_tuned_sbert: bool = False
    use_tuned_cross_encoder: bool = False

    # Retrieval parameters
    sbert_top_k: int = 10
    rerank_top_k: int = 5
    embedding_dim: int = 384

    # ── Batch sizes (optimized for RTX 3060 12GB) ──
    encode_batch_size: int = int(os.getenv("ENCODE_BATCH_SIZE", "1024"))
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "256"))

    # ── Performance optimizations ──
    device: str = DEVICE
    use_fp16: bool = DEVICE == "cuda"
    use_faiss: bool = True
    faiss_nprobe: int = 8


@dataclass
class GateConfig:
    """Auto-match gate thresholds (all configurable)."""
    min_score: float = 0.90
    margin: float = 0.10
    kickoff_window_minutes: int = 30
    tight_kickoff_minutes: int = 15

    sensitive_categories: list = field(default_factory=lambda: [
        "WOMEN", "U23", "U21", "U20", "U19", "U18", "U17",
        "RESERVES", "B-TEAM", "YOUTH", "AMATEUR"
    ])
    block_sensitive_auto_match: bool = True

    min_team_similarity: float = 0.25
    team_sim_weight: float = 0.70


@dataclass
class TrainingConfig:
    """Training hyperparameters (batch sizes optimized for RTX 3060 12GB)."""
    sbert_epochs: int = int(os.getenv("SBERT_EPOCHS", "3"))
    sbert_lr: float = float(os.getenv("SBERT_LR", "2e-5"))
    sbert_warmup_ratio: float = 0.1
    sbert_batch_size: int = int(os.getenv("SBERT_BATCH_SIZE", "128"))

    ce_epochs: int = int(os.getenv("CE_EPOCHS", "3"))
    ce_lr: float = float(os.getenv("CE_LR", "2e-5"))
    ce_batch_size: int = int(os.getenv("CE_BATCH_SIZE", "64"))

    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2"))
    dataloader_workers: int = int(os.getenv(
        "DATALOADER_WORKERS",
        "4" if DEVICE == "cuda" else "0",
    ))

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    max_hard_negatives_per_positive: int = 4
    min_hard_negative_score: float = 0.3

    use_amp: bool = DEVICE == "cuda"


@dataclass
class FeedbackAPIConfig:
    """CSE feedback API settings for self-training."""
    prod_url: str = os.getenv(
        "FEEDBACK_API_PROD_URL",
        "https://sports-bet-api.allinsports.online/api/matches/get-ai-mapping-feedback-v2",
    )
    local_url: str = os.getenv(
        "FEEDBACK_API_LOCAL_URL",
        "http://localhost:8010/api/matches/get-ai-mapping-feedback-v2",
    )
    use_local: bool = os.getenv("FEEDBACK_API_USE_LOCAL", "false").lower() == "true"
    default_platform: str = os.getenv("FEEDBACK_API_PLATFORM", "ODDSPORTAL")
    request_timeout: int = int(os.getenv("FEEDBACK_API_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("FEEDBACK_API_MAX_RETRIES", "3"))

    min_feedback_for_training: int = int(os.getenv("MIN_FEEDBACK_FOR_TRAINING", "50"))
    auto_reload_after_training: bool = os.getenv("AUTO_RELOAD_AFTER_TRAINING", "true").lower() == "true"

    @property
    def url(self) -> str:
        return self.local_url if self.use_local else self.prod_url


PROVIDER_REGISTRY = {
    "ODDSPORTAL": {
        "suffix": "/get-odds-portal-matches-with-odds",
        "paginated": True,
        "data_key": "rows",
    },
    "FLASHSCORE": {
        "suffix": "/get-flashscore-to-bet365-unmapped-matches",
        "paginated": False,
        "data_key": "flashScoreMatches",
        "bet365_key": "bet365Matches",
    },
    "SOFASCORE": {
        "suffix": "/get-sofascore-to-bet365-unmapped-matches",
        "paginated": False,
        "data_key": "sofaScoreMatches",
        "bet365_key": "bet365Matches",
    },
    "SBO": {
        "suffix": "/get-sbo-to-bet365-unmapped-matches",
        "paginated": False,
        "data_key": "sboMatches",
        "bet365_key": "bet365Matches",
    },
}


@dataclass
class APIEndpoints:
    """All external API URLs — every URL is configurable via env."""
    matches_base_url: str = os.getenv(
        "MATCHES_API_BASE_URL",
        "https://sports-bet-api.allinsports.online/api/matches",
    )
    store_results_url: str = os.getenv(
        "STORE_RESULTS_URL",
        "https://sports-bet-api.allinsports.online/api/matches/store-ai-mapping",
    )

    enabled_providers: list = field(default_factory=lambda: [
        p.strip().upper()
        for p in os.getenv("ENABLED_PROVIDERS", "ODDSPORTAL").split(",")
        if p.strip()
    ])

    @property
    def bet365_endpoint(self) -> str:
        return f"{self.matches_base_url}/get-bet365-matches-with-odds"

    @property
    def oddsportal_endpoint(self) -> str:
        return f"{self.matches_base_url}/get-odds-portal-matches-with-odds"

    def get_provider_endpoint(self, provider_name: str) -> str:
        """Get the full API endpoint URL for a given provider name."""
        name = provider_name.upper()
        info = PROVIDER_REGISTRY.get(name)
        if not info:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Known: {list(PROVIDER_REGISTRY.keys())}"
            )
        return f"{self.matches_base_url}{info['suffix']}"

    def get_provider_info(self, provider_name: str) -> dict:
        """Get full provider config including response format metadata."""
        name = provider_name.upper()
        info = PROVIDER_REGISTRY.get(name)
        if not info:
            raise ValueError(f"Unknown provider: {provider_name}")
        return {
            "name": name,
            "endpoint": f"{self.matches_base_url}{info['suffix']}",
            "paginated": info.get("paginated", True),
            "data_key": info.get("data_key", "rows"),
            "bet365_key": info.get("bet365_key"),
        }

    def get_active_providers(self) -> list:
        """Return list of provider info dicts for each enabled provider."""
        return [
            self.get_provider_info(name)
            for name in self.enabled_providers
            if name in PROVIDER_REGISTRY
        ]


@dataclass
class OutputConfig:
    """Controls how inference results are stored / delivered."""
    save_to_file: bool = os.getenv("SAVE_OUTPUT_TO_FILE", "true").lower() == "true"
    push_to_api: bool = os.getenv("PUSH_RESULTS_TO_API", "false").lower() == "true"
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
    push_workers: int = int(os.getenv("PUSH_WORKERS", "10"))


@dataclass
class SchedulerConfig:
    """Scheduler settings — all configurable via environment variables."""
    interval_minutes: int = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", "45"))
    platform: str = os.getenv("SCHEDULER_PLATFORM", "ODDSPORTAL")
    enable_training: bool = os.getenv("SCHEDULER_ENABLE_TRAINING", "true").lower() == "true"
    enable_inference: bool = os.getenv("SCHEDULER_ENABLE_INFERENCE", "true").lower() == "true"
    use_local_feedback_api: bool = os.getenv("SCHEDULER_USE_LOCAL_FEEDBACK_API", "false").lower() == "true"
    max_retries_per_phase: int = int(os.getenv("SCHEDULER_MAX_RETRIES", "2"))


@dataclass
class AppConfig:
    """Application-level settings."""
    data_dir: str = "./data"
    training_data_dir: str = "./training_data"
    models_dir: str = "./models"
    reports_dir: str = "./reports"
    logs_dir: str = "./logs"
    accuracy_log: str = "./logs/accuracy_history.jsonl"

    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "match_mapping")

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    model: ModelConfig = field(default_factory=ModelConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feedback_api: FeedbackAPIConfig = field(default_factory=FeedbackAPIConfig)
    endpoints: APIEndpoints = field(default_factory=APIEndpoints)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


CONFIG = AppConfig()

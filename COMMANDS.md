# AI Match Mapping Engine — Commands Reference

All commands to install, configure, run, and manage the application.

---

## 1. Setup & Installation

```bash
# Clone / navigate to project
cd match-mapping-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your settings (API URLs, providers, etc.)
```

---

## 2. Environment Configuration

Edit `.env` to control all runtime behavior. Key variables:

| Variable | Default | Description |
|---|---|---|
| `MATCHES_API_BASE_URL` | `https://sports-bet-api.allinsports.online/api/matches` | Base URL for all match data APIs |
| `STORE_RESULTS_URL` | `.../store-ai-mapping` | API endpoint to push AI mapping results |
| `ENABLED_PROVIDERS` | `ODDSPORTAL` | Comma-separated list of providers to map against Bet365. Options: `ODDSPORTAL`, `FLASHSCORE`, `SOFASCORE`, `SBO` |
| `SAVE_OUTPUT_TO_FILE` | `true` | Save results to `data/ai_suggested_mappings.json` |
| `PUSH_RESULTS_TO_API` | `false` | Push each result to `STORE_RESULTS_URL` |
| `CONFIDENCE_THRESHOLD` | `0.90` | Minimum confidence score to include in output |
| `PUSH_WORKERS` | `10` | Number of parallel threads for API push |
| `SCHEDULER_INTERVAL_MINUTES` | `45` | Minutes between automated scheduler cycles |
| `SCHEDULER_ENABLE_TRAINING` | `true` | Enable self-training phase in scheduler |
| `SCHEDULER_ENABLE_INFERENCE` | `true` | Enable inference phase in scheduler |

---

## 3. Automated Scheduler (Recommended for Production)

The scheduler runs the full pipeline in a loop: **self-train → fetch data → inference → push results**.

```bash
# Run with default settings (loops every 45 minutes)
python scripts/scheduler.py

# Run a single cycle and exit
python scripts/scheduler.py --run-once

# Skip the self-training phase (only fetch + inference)
python scripts/scheduler.py --skip-training

# Skip inference phase (only self-training)
python scripts/scheduler.py --skip-inference

# Override interval (run every 30 minutes)
python scripts/scheduler.py --interval 30

# Combine flags
python scripts/scheduler.py --run-once --skip-training

# Override interval via environment variable
SCHEDULER_INTERVAL_MINUTES=60 python scripts/scheduler.py
```

**Stop gracefully:** Press `Ctrl+C` — the scheduler finishes the current cycle before exiting.

---

## 4. API Pipeline (One-Time Run)

Fetches live data from all enabled providers, runs AI inference, and outputs results. Same as one scheduler cycle but without the training phase or looping.

```bash
# Run full pipeline (fetch from APIs + inference)
python scripts/api_pipeline.py

# Reuse previously fetched data (skip API calls, use cached data/*.json files)
python scripts/api_pipeline.py --skip-fetch
```

---

## 5. FastAPI Server

Starts the REST API server for real-time predictions and feedback ingestion.

```bash
# Start the API server (default: http://0.0.0.0:8000)
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Start with auto-reload for development
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Override host/port via environment
API_HOST=127.0.0.1 API_PORT=9000 uvicorn api.server:app --host 127.0.0.1 --port 9000
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Get Top-5 candidates for a single OP match |
| `POST` | `/predict/batch` | Batch predictions for multiple OP matches |
| `POST` | `/index/refresh` | Re-index the Bet365 candidate pool |
| `POST` | `/feedback` | Ingest a human decision (MATCH/SWAPPED/NO_MATCH) |
| `POST` | `/self-train` | Trigger self-training from CSE feedback |
| `POST` | `/models/reload` | Hot-reload tuned models without restart |
| `GET` | `/health` | Health check with system status |
| `GET` | `/metrics` | Current feedback and training pair stats |
| `GET` | `/training-pairs` | Export training pairs for inspection |
| `GET` | `/docs` | Interactive Swagger UI documentation |

---

## 6. Self-Training from CSE Feedback

Fetches human feedback from the CSE team API, converts it to training pairs, and retrains the models.

```bash
# Self-train using production feedback API
python scripts/self_train_pipeline.py --platform ODDSPORTAL

# Use local feedback API instead
python scripts/self_train_pipeline.py --platform ODDSPORTAL --use-local

# Dry run — fetch and process feedback but don't train
python scripts/self_train_pipeline.py --platform ODDSPORTAL --dry-run

# Train only SBERT (skip Cross-Encoder)
python scripts/self_train_pipeline.py --platform ODDSPORTAL --sbert-only

# Train only Cross-Encoder (skip SBERT)
python scripts/self_train_pipeline.py --platform ODDSPORTAL --ce-only

# Custom output paths for trained models
python scripts/self_train_pipeline.py --platform ODDSPORTAL \
    --sbert-output models/my_sbert \
    --ce-output models/my_ce

# Skip auto-reload of models after training
python scripts/self_train_pipeline.py --platform ODDSPORTAL --no-reload
```

---

## 7. Training from Labeled Data Files

Train models from pre-existing labeled records (JSON files).

```bash
# Train both models from labeled data
python scripts/train_models.py --data-file data/labeled_records.json

# Train only SBERT
python scripts/train_models.py --data-file data/labeled_records.json --sbert-only

# Train only Cross-Encoder
python scripts/train_models.py --data-file data/labeled_records.json --ce-only

# Custom output paths
python scripts/train_models.py --data-file data/labeled_records.json \
    --sbert-output models/sbert_v2 \
    --ce-output models/ce_v2

# Override epochs and batch size
python scripts/train_models.py --data-file data/labeled_records.json \
    --epochs 5 --batch-size 64

# Skip accuracy tracking (pre/post comparison)
python scripts/train_models.py --data-file data/labeled_records.json \
    --no-accuracy-tracking
```

---

## 8. Prepare Training Data from MongoDB Exports

Parse MongoDB JSON exports into labeled records and training pairs.

```bash
# Process all JSON files in training_data/ folder
python scripts/prepare_data.py

# Process a specific folder
python scripts/prepare_data.py --input training_data/

# Process a single file
python scripts/prepare_data.py --input training_data/my_export.json

# Custom output directory
python scripts/prepare_data.py --input training_data/ --output-dir data/
```

**Output files:** `data/labeled_records.json`, `data/training_pairs.json`, `data/b365_pool.json`, `data/prepare_stats.json`

Place your MongoDB JSON export files in the `training_data/` folder before running.

---

## 9. Accuracy Tracking & Evaluation

Measure model performance, track improvement history, and compare training runs.

```bash
# Evaluate current models on labeled data
python scripts/evaluate_accuracy.py --data-file data/training_pairs.json

# Evaluate specific tuned models
python scripts/evaluate_accuracy.py --data-file data/training_pairs.json \
    --sbert-path models/sbert_tuned_20260220 \
    --ce-path models/ce_tuned_20260220

# View full accuracy history (all past training runs)
python scripts/evaluate_accuracy.py --history

# Compare a specific training run (before vs after)
python scripts/evaluate_accuracy.py --compare run_20260220_140000
```

---

## 10. Demo Pipeline

Run a full end-to-end demo with synthetic data to verify the system works.

```bash
python scripts/demo_pipeline.py
```

This generates synthetic OP + B365 matches, runs inference, ingests feedback, trains models, evaluates, and demonstrates hot-swap — useful for verifying a new installation.

---

## 11. Provider Management

Control which providers are mapped against Bet365 via the `ENABLED_PROVIDERS` environment variable.

```bash
# Map only OddsPortal (default)
ENABLED_PROVIDERS=ODDSPORTAL python scripts/scheduler.py --run-once

# Map all four providers
ENABLED_PROVIDERS=ODDSPORTAL,FLASHSCORE,SOFASCORE,SBO python scripts/scheduler.py --run-once

# Map only FlashScore and SBO
ENABLED_PROVIDERS=FLASHSCORE,SBO python scripts/scheduler.py --run-once
```

Or set it permanently in `.env`:
```
ENABLED_PROVIDERS=ODDSPORTAL,FLASHSCORE,SOFASCORE,SBO
```

**Available providers:**

| Provider | API Endpoint Suffix |
|---|---|
| `ODDSPORTAL` | `/get-odds-portal-matches-with-odds` |
| `FLASHSCORE` | `/get-flashscore-to-bet365-unmapped-matches` |
| `SOFASCORE` | `/get-sofascore-to-bet365-unmapped-matches` |
| `SBO` | `/get-sbo-to-bet365-unmapped-matches` |

To add a new provider, add one line to `PROVIDER_REGISTRY` in `config/settings.py`.

---

## 12. Quick Reference

| What you want to do | Command |
|---|---|
| **Run everything automatically** | `python scripts/scheduler.py` |
| **Run one cycle and exit** | `python scripts/scheduler.py --run-once` |
| **Run inference only (no training)** | `python scripts/scheduler.py --run-once --skip-training` |
| **Run standalone inference** | `python scripts/api_pipeline.py` |
| **Start REST API server** | `uvicorn api.server:app --host 0.0.0.0 --port 8000` |
| **Self-train from CSE feedback** | `python scripts/self_train_pipeline.py --platform ODDSPORTAL` |
| **Train from labeled JSON** | `python scripts/train_models.py --data-file data/labeled_records.json` |
| **Prepare MongoDB data** | `python scripts/prepare_data.py` |
| **Check accuracy history** | `python scripts/evaluate_accuracy.py --history` |
| **Run demo** | `python scripts/demo_pipeline.py` |

# AI Match Mapping Engine — Complete Guide

Step-by-step instructions for training, running, and maintaining the model.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Training on Old Data (MongoDB Exports)](#4-training-on-old-data-mongodb-exports)
5. [Training from CSE Feedback (Self-Training)](#5-training-from-cse-feedback-self-training)
6. [Running the Inference Pipeline](#6-running-the-inference-pipeline)
7. [Running the API Server](#7-running-the-api-server)
8. [Deploying Trained Models](#8-deploying-trained-models)
9. [Docker Deployment](#9-docker-deployment)
10. [Evaluation](#10-evaluation)
11. [Accuracy Tracking & Comparison](#11-accuracy-tracking--comparison)
12. [Automated Scheduler](#12-automated-scheduler)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

- **Python** 3.10+
- **pip** or **conda** for package management
- **8 GB+ RAM** (models + B365 pool embeddings)
- **GPU optional** (MPS on Mac, CUDA on Linux/Windows — falls back to CPU)

---

## 2. Project Structure

```
match-mapping-engine/
├── training_data/              ← PUT YOUR JSON FILES HERE FOR TRAINING
│   ├── matches_export.json     ← MongoDB exports (array of documents)
│   └── ...                     ← Multiple files supported
├── data/                       ← Generated training data + inference output
│   ├── labeled_records.json    ← Parsed records (from prepare_data.py)
│   ├── training_pairs.json     ← Training pairs (anchor + candidate + label)
│   ├── b365_pool.json          ← Unique B365 matches for indexing
│   └── ai_suggested_mappings.json  ← Inference output
├── models/                     ← Fine-tuned models saved here
│   ├── sbert_tuned_YYYYMMDD/
│   └── ce_tuned_YYYYMMDD/
├── scripts/
│   ├── prepare_data.py         ← Step 1: Parse JSON → training data
│   ├── train_models.py         ← Step 2: Fine-tune SBERT + Cross-Encoder
│   ├── self_train_pipeline.py  ← Self-train from CSE API feedback
│   ├── evaluate_accuracy.py   ← Standalone accuracy evaluation & comparison
│   ├── scheduler.py           ← Automated pipeline (train → fetch → infer loop)
│   ├── api_pipeline.py         ← Live inference pipeline
│   └── demo_pipeline.py        ← Demo with synthetic data
├── core/
│   ├── inference.py            ← SBERT retrieval + CE reranking
│   ├── normalizer.py           ← Text normalization + team aliases
│   ├── models.py               ← Data models
│   └── feedback.py             ← Feedback ingestion + training pairs
├── training/
│   └── trainer.py              ← SBERT + CE training logic
├── api/
│   └── server.py               ← FastAPI server
├── evaluation/
│   ├── evaluator.py            ← Offline evaluation (Recall, Precision, etc.)
│   └── accuracy_tracker.py     ← Accuracy logging, scoring, comparison reports
├── logs/                       ← Auto-created by accuracy tracker
│   ├── accuracy_history.jsonl  ← Append-only accuracy log (one JSON per line)
│   └── comparison_run_*.txt    ← Before/after comparison reports
├── config/
│   └── settings.py             ← All configurable thresholds
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── GUIDE.md                    ← This file
```

---

## 3. Installation

### Option A: Virtual Environment (Recommended)

```bash
cd match-mapping-engine

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using existing venv

```bash
./venv/bin/pip install -r requirements.txt
```

### Verify installation

```bash
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

---

## 4. Training on Old Data (MongoDB Exports)

This is how you train the model using your existing back-office mapping data exported from MongoDB.

### Step 4.1: Prepare your JSON files

Export your MongoDB collection as JSON. Each document should be an OddsPortal match with an embedded `mappedData` field for the Bet365 match.

**Expected document format:**

```json
{
  "id": "QPF2xRgl",
  "platform": "ODDSPORTAL",
  "sport": "Basketball",
  "home_team": "Frayles de Guasave",
  "away_team": "Rayos de Hermosillo",
  "league": {
    "name": "Cibacopa",
    "league_name_en": "Cibacopa"
  },
  "commence_time": 1771469719,
  "isTeamSwitch": false,
  "mappedData": {
    "id": "189929499",
    "platform": "BET365",
    "sport": "Basketball",
    "home_team": "Frayles de Guasave",
    "away_team": "Rayos de Hermosillo",
    "league": {
      "name": "Mexico CIBACOPA"
    },
    "commence_time": 1771468200,
    "entry_type": "MANUAL",
    "lastUpdatedBy": "admin"
  }
}
```

**How decisions are derived:**
- `mappedData` exists with a valid `id` → **MATCH** (or **SWAPPED** if `isTeamSwitch` is true)
- `mappedData` is null/missing/empty → **NO_MATCH**

### Step 4.2: Place files in training_data/ folder

```bash
# Copy your MongoDB JSON exports into the training_data folder
cp /path/to/your/mongodb_export.json training_data/

# You can place multiple files — all will be loaded
cp /path/to/export_part1.json training_data/
cp /path/to/export_part2.json training_data/
```

**Supported formats:**
- JSON array: `[ {doc1}, {doc2}, ... ]`
- Single document: `{ ... }`
- JSONL (one document per line)
- MongoDB export with `$oid` / `$date` wrappers
- Multiple files in the folder (all `.json` files are loaded)

### Step 4.3: Run data preparation

```bash
# Default: reads from training_data/, outputs to data/
python scripts/prepare_data.py

# Custom input path
python scripts/prepare_data.py --input training_data/my_export.json

# Custom output directory
python scripts/prepare_data.py --input training_data/ --output-dir data/

# Only use MANUAL entries (higher quality labels, skip AUTO)
python scripts/prepare_data.py --manual-only
```

**Output files created in `data/`:**

| File | Description |
|------|-------------|
| `labeled_records.json` | Standardized records (used by train_models.py) |
| `training_pairs.json` | Direct training pairs with labels |
| `b365_pool.json` | Unique B365 matches for indexing |
| `prepare_stats.json` | Statistics summary |

**Example output:**

```
======================================================================
  DATA PREPARATION — MongoDB JSON → Training Data
======================================================================

Step 1: Loading JSON files from training_data/...
  Loaded 10000 raw documents

Step 2: Parsing MongoDB documents...
  Parsed: 9850 records (150 errors)
  Decisions: {"MATCH": 8500, "SWAPPED": 350, "NO_MATCH": 1000}

Step 3: Generating training pairs...
  Training pairs: 9550
    Positives:      8850
    Hard negatives: 700

Step 4: Extracting B365 match pool...
  Unique B365 matches: 8200

Step 5: Saving to data/...
  data/labeled_records.json (4500.2 KB)
  data/training_pairs.json (3200.1 KB)
  data/b365_pool.json (1800.5 KB)
  data/prepare_stats.json (0.4 KB)
```

### Step 4.4: Train the models

```bash
# Train both SBERT + Cross-Encoder (recommended)
python scripts/train_models.py --data-file data/labeled_records.json

# Train only the SBERT bi-encoder (faster retrieval)
python scripts/train_models.py --data-file data/labeled_records.json --sbert-only

# Train only the Cross-Encoder reranker (better scoring)
python scripts/train_models.py --data-file data/labeled_records.json --ce-only

# Custom training parameters
python scripts/train_models.py \
  --data-file data/labeled_records.json \
  --epochs 5 \
  --batch-size 16
```

**Training takes:**
- SBERT: ~5-15 minutes (depending on data size and hardware)
- Cross-Encoder: ~10-30 minutes
- Both: ~15-45 minutes

**Output:**
- `models/sbert_tuned_YYYYMMDD_HHMMSS/` — fine-tuned SBERT
- `models/ce_tuned_YYYYMMDD_HHMMSS/` — fine-tuned Cross-Encoder

### Step 4.5: Deploy the trained models

See [Section 8: Deploying Trained Models](#8-deploying-trained-models).

---

## 5. Training from CSE Feedback (Self-Training)

After the model is running in production, the CSE team reviews AI suggestions and provides feedback. This feedback is used to continuously improve the models.

### Feedback types and their training effect

| CSE Feedback | Internal Decision | Training Effect | Score Change |
|---|---|---|---|
| **Correct** | MATCH | Positive pair (label=1.0) | 9.0 → 9.5 |
| **Not correct** | NO_MATCH | Hard negative (label=0.0) | 8.5 → 7.0 |
| **Need to swap** | SWAPPED | Positive (swapped) + negative (original) | Learns correct order |
| **Not Sure** | — | Skipped | No signal |

### Option A: CLI Script

```bash
# Dry run first (fetch feedback, preview what would happen)
python scripts/self_train_pipeline.py --platform ODDSPORTAL --dry-run

# Train using production feedback API
python scripts/self_train_pipeline.py --platform ODDSPORTAL

# Train using local API (localhost:8010)
python scripts/self_train_pipeline.py --platform ODDSPORTAL --use-local

# Train only one model
python scripts/self_train_pipeline.py --platform ODDSPORTAL --sbert-only
python scripts/self_train_pipeline.py --platform ODDSPORTAL --ce-only

# Merge with old training data
python scripts/self_train_pipeline.py \
  --platform ODDSPORTAL \
  --include-existing data/labeled_records.json

# Override params
python scripts/self_train_pipeline.py \
  --platform ODDSPORTAL \
  --epochs 5 \
  --batch-size 16 \
  --min-feedback 10
```

### Option B: API Endpoint (for automation)

```bash
# Start the API server first
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Dry run
curl -X POST http://localhost:8000/self-train \
  -H "Content-Type: application/json" \
  -d '{"platform": "ODDSPORTAL", "dry_run": true}'

# Full training (runs in background, auto-reloads models)
curl -X POST http://localhost:8000/self-train \
  -H "Content-Type: application/json" \
  -d '{"platform": "ODDSPORTAL", "auto_reload": true}'
```

---

## 6. Running the Inference Pipeline

The live pipeline fetches data from APIs, runs AI inference, and saves results.

### Using cached data (skip API fetch)

```bash
python scripts/api_pipeline.py --skip-fetch
```

### Full pipeline (fetch + inference)

```bash
python scripts/api_pipeline.py
```

### Demo with synthetic data

```bash
python scripts/demo_pipeline.py
```

**Output:** `data/ai_suggested_mappings.json` — contains AI-suggested match mappings with confidence scores.

---

## 7. Running the API Server

### Start the server

```bash
# Direct
uvicorn api.server:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Using the venv
./venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Key API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get Top-5 candidates for one OP match |
| `/predict/batch` | POST | Batch predictions |
| `/index/refresh` | POST | Upload + index B365 matches |
| `/feedback` | POST | Submit human decision |
| `/self-train` | POST | Trigger self-training from CSE feedback |
| `/models/reload` | POST | Hot-swap models |
| `/health` | GET | System health check |
| `/metrics` | GET | Feedback + training stats |

### API docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 8. Deploying Trained Models

After training completes, models are saved in the `models/` folder. To activate them:

### Option A: Via API (hot-reload, no restart needed)

```bash
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{
    "use_tuned_sbert": true,
    "use_tuned_cross_encoder": true,
    "tuned_sbert_path": "models/sbert_tuned_20260220_130000",
    "tuned_cross_encoder_path": "models/ce_tuned_20260220_130000"
  }'
```

### Option B: Via config (permanent, requires restart)

Edit `config/settings.py`:

```python
@dataclass
class ModelConfig:
    tuned_sbert_path: str = "models/sbert_tuned_20260220_130000"
    tuned_cross_encoder_path: str = "models/ce_tuned_20260220_130000"
    use_tuned_sbert: bool = True
    use_tuned_cross_encoder: bool = True
```

### Rollback to base models

```bash
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{"use_tuned_sbert": false, "use_tuned_cross_encoder": false}'
```

---

## 9. Docker Deployment

### Build and run

```bash
docker compose up -d
```

### With MongoDB

The `docker-compose.yml` includes a MongoDB service. To use an external MongoDB:

```bash
export MONGO_URI="mongodb://your-host:27017"
docker compose up -d
```

---

## 10. Evaluation

### Run offline evaluation

```bash
python scripts/evaluate.py \
  --test-data data/labeled_records.json \
  --sbert-model models/sbert_tuned_20260220 \
  --ce-model models/ce_tuned_20260220
```

### Success metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | >= 90% | Correct B365 match appears in Top-5 |
| Recall@10 | >= 95% | Correct B365 match appears in Top-10 |
| AUTO_MATCH Precision | >= 98% | Auto-confirmed matches are correct |
| No-Match FP Rate | < 2% | System doesn't auto-match when no match exists |
| Automation Rate | Increasing | More AUTO_MATCH over time |

---

## 11. Accuracy Tracking & Comparison

The system automatically tracks model accuracy before and after every training
run so you can verify improvement objectively.

### How it works

1. **Before training** — the test split is scored with the current SBERT and
   Cross-Encoder models.  Metrics are recorded to `logs/accuracy_history.jsonl`.
2. **After training** — the same test split is scored with the newly-trained
   models.  A second entry is appended to the log.
3. **Comparison report** — a human-readable diff is printed to the terminal
   and saved to `logs/comparison_run_*.txt`.

All of this happens automatically when you run `train_models.py` or
`self_train_pipeline.py`.

### Metrics tracked

| Metric | Description | Goal |
|--------|-------------|------|
| **CE Positive Avg Score** | Mean cross-encoder score on true-match pairs | Higher after training |
| **CE Negative Avg Score** | Mean cross-encoder score on non-match pairs | Lower after training |
| **CE Score Gap** | Positive avg minus Negative avg (separation quality) | Wider is better |
| **CE Binary Accuracy** | % of pairs correctly classified at 0.5 threshold | Higher is better |
| **CE AUC-ROC** | Area under ROC curve (ranking quality) | Higher is better |
| **SBERT Positive Avg Sim** | Mean cosine similarity for true-match pairs | Higher after training |
| **SBERT Negative Avg Sim** | Mean cosine similarity for non-match pairs | Lower after training |
| **SBERT Similarity Gap** | Positive avg minus Negative avg | Wider is better |

### View accuracy history

```bash
python scripts/evaluate_accuracy.py --history
```

Output:

```
====================================================================================================
  MODEL ACCURACY HISTORY
====================================================================================================

  Timestamp              Run Type         Run ID                   CE Acc   CE AUC   CE Gap  SBERT Gap
  ----------------------------------------------------------------------------------------------------
  2026-02-20T14:00:00    pre_training     run_20260220_140000      0.8234   0.8567   0.3256     0.3333
  2026-02-20T14:30:00    post_training    run_20260220_140000      0.8921   0.9234   0.5302     0.4343
  2026-02-21T10:15:00    pre_training     run_20260221_101500      0.8921   0.9234   0.5302     0.4343
  2026-02-21T10:45:00    post_training    run_20260221_101500      0.9112   0.9456   0.5890     0.4701
```

### Compare a specific training run

```bash
python scripts/evaluate_accuracy.py --compare run_20260220_140000
```

This prints the full before/after comparison with deltas and a verdict
(IMPROVED / REGRESSED / MIXED / NO CHANGE).

### Evaluate a model manually

You can evaluate any model against labeled data without running training:

```bash
# Evaluate current configured models
python scripts/evaluate_accuracy.py --data-file data/training_pairs.json

# Evaluate specific tuned models
python scripts/evaluate_accuracy.py --data-file data/training_pairs.json \
    --sbert-path models/sbert_tuned_20260220 \
    --ce-path models/ce_tuned_20260220
```

### Disable accuracy tracking (faster training)

If you want to skip pre/post evaluation for faster training runs:

```bash
python scripts/train_models.py --data-file data/labeled_records.json --no-accuracy-tracking
```

### Log file location

All accuracy data is stored in `logs/accuracy_history.jsonl` (one JSON object
per line, append-only).  You can parse this file with any JSON tool:

```bash
# Count entries
wc -l logs/accuracy_history.jsonl

# Pretty-print latest entry
tail -1 logs/accuracy_history.jsonl | python -m json.tool
```

---

## 12. Automated Scheduler

The scheduler runs the entire pipeline in a continuous loop on a configurable
interval (default: every 45 minutes).

### What it does each cycle

```
CYCLE (every N minutes)
  │
  ├── Phase 1: SELF-TRAIN
  │   Fetch CSE feedback → train models → reload into engine
  │
  ├── Phase 2: FETCH DATA
  │   Fetch Bet365 + OddsPortal matches concurrently
  │
  └── Phase 3: INFERENCE
      Index B365 → batch inference on OP → save results → (optional) push to API
```

### Quick start

```bash
# Run with default 45-minute interval
python scripts/scheduler.py

# Override interval via env
SCHEDULER_INTERVAL_MINUTES=30 python scripts/scheduler.py

# Single run (no loop)
python scripts/scheduler.py --run-once

# Skip training, only fetch + infer
python scripts/scheduler.py --skip-training

# Skip inference, only train
python scripts/scheduler.py --skip-inference
```

### Environment variables

All scheduler settings come from environment variables. Copy `.env.example`
to `.env` and edit:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_INTERVAL_MINUTES` | `45` | Minutes between each cycle |
| `SCHEDULER_PLATFORM` | `ODDSPORTAL` | Platform identifier |
| `SCHEDULER_ENABLE_TRAINING` | `true` | Run Phase 1 (self-training) |
| `SCHEDULER_ENABLE_INFERENCE` | `true` | Run Phase 2+3 (fetch + infer) |
| `SCHEDULER_USE_LOCAL_FEEDBACK_API` | `false` | Use `localhost:8010` for feedback |
| `SCHEDULER_PUSH_RESULTS` | `false` | POST results to external store API |
| `SCHEDULER_STORE_RESULTS_URL` | `https://...store-ai-mapping` | URL for pushing results |
| `SCHEDULER_CONFIDENCE_THRESHOLD` | `0.90` | Minimum confidence for output |
| `MATCHES_API_BASE_URL` | `https://...allinsports.online/api/matches` | Base URL for data APIs |

### Graceful shutdown

Press `Ctrl+C` (SIGINT) or send `SIGTERM` — the scheduler finishes the current
cycle and exits cleanly. It will not start a new cycle after receiving the signal.

### Cycle reports

Each cycle writes a JSON report to `logs/cycle_N_TIMESTAMP.json` with the
status of each phase, timings, and error details (if any). These accumulate
over time for operational auditing.

### Running in production

```bash
# Background with nohup
nohup python scripts/scheduler.py > logs/scheduler.log 2>&1 &

# Or via Docker (add to docker-compose.yml)
# See Section 9 for Docker deployment

# Or as a systemd service
# [Unit]
# Description=AI Match Mapping Scheduler
# After=network.target
#
# [Service]
# WorkingDirectory=/path/to/match-mapping-engine
# ExecStart=/path/to/venv/bin/python scripts/scheduler.py
# Restart=always
# EnvironmentFile=/path/to/match-mapping-engine/.env
#
# [Install]
# WantedBy=multi-user.target
```

---

## 13. Troubleshooting

### "Precision 'float16' is not supported"

Your `sentence-transformers` version doesn't support FP16 precision. This has been fixed — the code no longer passes the `precision` parameter. If you still see this error, pull the latest code.

### "No JSON files found in training_data/"

Place your MongoDB JSON exports in the `training_data/` folder. Files must have `.json` extension.

### "ABORTING: Label contamination detected!"

The same (anchor, candidate) text pair appears as both a positive and a negative in your training data. This usually means a data quality issue in your exports. Check for duplicate records with conflicting decisions.

### Out of memory during training

Reduce batch size:

```bash
python scripts/train_models.py --data-file data/labeled_records.json --batch-size 8
```

### Out of memory during inference

Reduce `encode_batch_size` and `rerank_batch_size` in `config/settings.py`:

```python
encode_batch_size: int = 256    # default 512
rerank_batch_size: int = 64     # default 128
```

### Models not improving after training

- Ensure you have enough training data (minimum ~100 labeled records)
- Increase epochs: `--epochs 5` or `--epochs 10`
- Make sure the data has both MATCH and NO_MATCH decisions
- Check `data/prepare_stats.json` for data distribution

### Slow inference

- Ensure FAISS is installed: `pip install faiss-cpu`
- The first run downloads models from HuggingFace (~160 MB). Subsequent runs use cached models.
- GPU (CUDA/MPS) significantly speeds up encoding.

---

## Complete Workflow Summary

```
              YOUR DATA
                 │
                 ▼
    ┌─────────────────────────┐
    │  training_data/*.json   │  ← Place MongoDB exports here
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  prepare_data.py        │  ← Parse JSON → training data
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  data/                  │
    │  ├─ labeled_records.json│  ← Standardized records
    │  ├─ training_pairs.json │  ← Training pairs
    │  └─ b365_pool.json      │  ← B365 match pool
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  train_models.py        │  ← Fine-tune SBERT + Cross-Encoder
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  models/                │
    │  ├─ sbert_tuned_*/      │  ← Fine-tuned SBERT
    │  └─ ce_tuned_*/         │  ← Fine-tuned Cross-Encoder
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Deploy via API         │
    │  POST /models/reload    │  ← Hot-swap to trained models
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Production Inference   │
    │  POST /predict          │  ← OP match → Top-5 B365 candidates
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  CSE Team Reviews       │  ← Correct / Not correct / Swap / Not sure
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  self_train_pipeline.py │  ← Re-train from feedback → loop back
    └─────────────────────────┘
```

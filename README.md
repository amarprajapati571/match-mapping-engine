# AI Match Mapping Engine

**SBERT Retrieval → Cross-Encoder Reranking → Active Learning**

An AI-assisted sports match mapping engine that links OddsPortal (OP) matches to Bet365 (B365) matches with high reliability and continuously improves through human feedback.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
│                                                                 │
│  OP Match ──→ Pre-Filter ──→ SBERT Retrieval ──→ Cross-Encoder  │
│               (sport +       (Top-10 by         Rerank (Top-5)  │
│                ±30min)        cosine sim)        + swap detect  │
│                                                      │          │
│                              ┌────────────────────────┘         │
│                              ▼                                  │
│                       Gate Evaluation                           │
│                    ┌──────────────────┐                         │
│                    │ MinScore ≥ 0.90  │                         │
│                    │ Margin  ≥ 0.10   │                         │
│                    │ Category match   │──→ AUTO_MATCH           │
│                    │ Kickoff ≤ 15min  │   or NEED_REVIEW        │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌──────────────────┐                    ┌──────────────────────┐
│  Admin UI        │                    │  Feedback Ingestion  |
│  (Top-5 review)  │───────────────────→│  MATCH / SWAPPED /   │
│                  │  human decision     │  NO_MATCH           │
└──────────────────┘                    │  → positive pairs    │
                                        │  → hard negatives    │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │  Model Tuning        │
                                        │  SBERT: MNR Loss     │
                                        │  CE: Binary Class.   │
                                        │  → feature flag swap │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │  Evaluation          │
                                        │  Recall@5/10         │
                                        │  AUTO_MATCH Prec.    │
                                        │  Drift monitoring    │
                                        └──────────────────────┘
```

---

## Free Models Used

| Component | Model | Source | Size |
|-----------|-------|--------|------|
| Bi-Encoder (retrieval) | `all-MiniLM-L6-v2` | HuggingFace / sentence-transformers | 80MB |
| Cross-Encoder (reranker) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace | 80MB |

Both models are free, open-source, and fine-tunable on your domain data.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run Demo (synthetic data)

```bash
python scripts/demo_pipeline.py
```

### 3. Start API Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 4. Automated Scheduler (every 45 min)

```bash
# Start the scheduler (runs: train → fetch → infer → repeat)
python scripts/scheduler.py

# Override interval via env
SCHEDULER_INTERVAL_MINUTES=30 python scripts/scheduler.py

# Single run, no loop
python scripts/scheduler.py --run-once
```

Configure via environment variables — see `.env.example`.

### 5. Docker

```bash
docker compose up -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Top-5 candidates + gate decision for one OP match |
| POST | `/predict/batch` | Batch predictions |
| POST | `/index/refresh` | Re-index B365 match pool |
| POST | `/feedback` | Ingest human decision (MATCH/SWAPPED/NO_MATCH) |
| POST | `/models/reload` | Hot-swap models (feature flag) |
| GET | `/health` | System health + model info |
| GET | `/metrics` | Feedback stats + training pair counts |
| GET | `/training-pairs` | Export training pairs for inspection |

### Example: Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "op_match": {
      "match_id": "OP_001",
      "platform": "OP",
      "sport": "soccer",
      "league": "Premier League",
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "kickoff": "2025-02-01T15:00:00"
    }
  }'
```

### Response

```json
{
  "mapping_suggestion_id": "uuid-...",
  "op_match_id": "OP_001",
  "candidates_top5": [
    {
      "rank": 1,
      "b365_match_id": "B365_042",
      "b365_home": "Arsenal",
      "b365_away": "Chelsea",
      "score": 0.9823,
      "time_diff_minutes": 2.0,
      "swapped": false
    },
    ...
  ],
  "gate_decision": "AUTO_MATCH",
  "gate_details": {
    "min_score_gate": true,
    "margin_gate": true,
    "category_gate": true,
    "kickoff_gate": true
  }
}
```

---

## Training Pipeline

### Train on Old Data (MongoDB Exports)

```bash
# 1. Place MongoDB JSON exports in training_data/
cp your_export.json training_data/

# 2. Convert to training format
python scripts/prepare_data.py

# 3. Train both models
python scripts/train_models.py --data-file data/labeled_records.json
```

### Self-Train from CSE Feedback

```bash
python scripts/self_train_pipeline.py --platform ODDSPORTAL
```

### Track Accuracy Improvement

Training automatically evaluates the model before and after, logging all
metrics to `logs/accuracy_history.jsonl`:

```bash
# View full accuracy history
python scripts/evaluate_accuracy.py --history

# Compare a specific training run (before vs after)
python scripts/evaluate_accuracy.py --compare run_20260220_140000

# Evaluate a model on labeled data without training
python scripts/evaluate_accuracy.py --data-file data/training_pairs.json
```

> **See [GUIDE.md](GUIDE.md) for full details** including all CLI options, data formats, and troubleshooting.

### Deploy with Feature Flag

```bash
# Toggle to tuned models
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{
    "use_tuned_sbert": true,
    "use_tuned_cross_encoder": true,
    "tuned_sbert_path": "models/sbert_tuned_20250215",
    "tuned_cross_encoder_path": "models/ce_tuned_20250215"
  }'

# Rollback to base models
curl -X POST http://localhost:8000/models/reload \
  -d '{"use_tuned_sbert": false, "use_tuned_cross_encoder": false}'
```

---

## Evaluation

### Expected Label Format

```json
[
  {
    "op_match_id": "OP_001",
    "op_league": "Premier League",
    "op_home": "Arsenal",
    "op_away": "Chelsea",
    "op_kickoff": "2025-02-01T15:00:00",
    "op_sport": "soccer",
    "b365_match_id": "B365_042",
    "b365_home": "Arsenal",
    "b365_away": "Chelsea",
    "b365_kickoff": "2025-02-01T14:58:00",
    "decision": "MATCH",
    "candidates": [
      {"b365_match_id": "B365_042", "b365_home": "Arsenal", "b365_away": "Chelsea"},
      {"b365_match_id": "B365_099", "b365_home": "Arsenal", "b365_away": "Liverpool"}
    ]
  }
]
```

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | ≥ 90% | Correct match in Top-5 |
| AUTO_MATCH Precision | ≥ 98% | Safe automation |
| No-Match FP Rate | < 2% | Don't auto-match when no match exists |
| Automation Rate | Increasing | More AUTO_MATCH over time |

---

## Project Structure

```
match-mapping-engine/
├── training_data/              # ← PUT YOUR JSON FILES HERE FOR TRAINING
├── data/                       # Generated training data + inference output
├── models/                     # Fine-tuned models saved here
├── api/
│   └── server.py               # FastAPI endpoints
├── config/
│   └── settings.py             # All configurable thresholds
├── core/
│   ├── models.py               # Pydantic data models
│   ├── normalizer.py           # Team aliases, category detection
│   ├── inference.py            # SBERT retrieval + CE reranking
│   └── feedback.py             # Feedback ingestion + training pairs
├── training/
│   └── trainer.py              # SBERT + CE fine-tuning
├── evaluation/
│   ├── evaluator.py            # Recall@K, Precision, drift monitoring
│   └── accuracy_tracker.py     # Accuracy logging & before/after comparison
├── logs/
│   └── accuracy_history.jsonl  # Append-only accuracy log
├── scripts/
│   ├── prepare_data.py         # Convert MongoDB JSON → training data
│   ├── train_models.py         # Fine-tune models CLI
│   ├── self_train_pipeline.py  # Self-train from CSE API feedback
│   ├── evaluate_accuracy.py    # Standalone accuracy evaluation
│   ├── scheduler.py            # Automated loop (train → fetch → infer)
│   ├── api_pipeline.py         # Live inference pipeline
│   └── demo_pipeline.py        # End-to-end demo
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── GUIDE.md                    # ← Complete training & usage instructions
└── README.md
```

> **See [GUIDE.md](GUIDE.md) for complete step-by-step instructions** on training, deployment, and troubleshooting.

---

## Key Design Decisions

1. **Two-stage retrieval**: SBERT (fast, broad) → Cross-Encoder (slow, precise). This scales to large B365 pools while maintaining accuracy.

2. **Swap detection**: Both normal and swapped OP text are scored against each B365 candidate. The higher score wins, and the `swapped` flag is set.

3. **Category tokens**: `[WOMEN]`, `[U23]`, etc. are prepended to text inputs. This gives the model explicit signal to distinguish Women's Arsenal from Men's Arsenal.

4. **Hard negative mining**: Unselected Top-5 candidates are the hardest negatives — they passed the retrieval filter but were rejected by humans. These are gold for training.

5. **Label contamination prevention**: The true positive is NEVER placed in the negative pool. Validated before every training run.

6. **Feature flags**: Models can be hot-swapped without restarting the API. Rollback is instant.

7. **Gate configurability**: All thresholds (min_score, margin, kickoff window) are in `config/settings.py` and can be tuned without code changes.

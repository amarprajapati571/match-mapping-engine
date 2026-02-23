# AI Match Mapping Engine — Complete Build Specification

> **Purpose**: Feed this file to Cursor IDE to build a production-ready AI match mapping system.
> **Input**: MongoDB JSON exports of OddsPortal matches with `mappedData` (Bet365) embedded.
> **Output**: Trained SBERT + Cross-Encoder models that auto-map OP → B365 matches.

---

## 1. ACTUAL DATA STRUCTURE (MongoDB Documents)

Each document is an **OddsPortal (OP)** match. If the back office team has mapped it, the `mappedData` field contains the **Bet365 (B365)** match.

### Root Document (OddsPortal Match)

```json
{
  "_id": { "$oid": "..." },
  "id": "QPF2xRgl",                          // OP match ID (use this)
  "platform": "ODDSPORTAL",                   // always "ODDSPORTAL"
  "sport": "Basketball",                       // sport name
  "sport_id": "4",                             // sport numeric ID
  "home_team": "Frayles de Guasave",           // OP home team
  "away_team": "Rayos de Hermosillo",          // OP away team
  "league": {
    "name": "Cibacopa",                        // OP league name
    "league_name_en": "Cibacopa",
    "country_en": null
  },
  "commence_time": 1771469719,                 // UNIX timestamp (seconds)
  "type": "PREMATCH",                          // PREMATCH or INPLAY
  "isTeamSwitch": false,                       // ★ SWAP flag for OP→B365
  "isTeamSwitchSBO": false,                    // swap flag for SBO (ignore)
  "is_mapping": true,                          // mapping is active
  "ismapped": false,                           // has been mapped (may be stale)
  "bet365_mapped_id": "189929499",             // B365 match ID (shortcut)
  "home_team_details": {                       // team metadata
    "_id": "...",
    "name_en": "Frayles de Guasave",
    "short_name_en": "Frayles de"
  },
  "away_team_details": {
    "_id": "...",
    "name_en": "Rayos de Hermosillo",
    "short_name_en": "Rayos de H"
  },
  "match_url": "https://www.oddsportal.com/...",
  "lastUpdatedBy": "admin",
  "created_at": { "$date": "2026-02-18T00:23:42.738Z" },
  "updatedAt": { "$date": "2026-02-20T06:18:28.508Z" },

  "mappedData": { ... }                        // ★ B365 MATCH (see below)
}
```

### `mappedData` Field (Bet365 Match)

```json
{
  "_id": { "$oid": "..." },
  "platform": "BET365",                        // always "BET365"
  "id": "189929499",                           // B365 match ID
  "sport": "Basketball",
  "sport_id": "4",
  "home_team": "Frayles de Guasave",           // B365 home team
  "away_team": "Rayos de Hermosillo",          // B365 away team
  "league": {
    "name": "Mexico CIBACOPA",                 // B365 league name (may differ!)
    "league_name_en": "Mexico CIBACOPA"
  },
  "commence_time": 1771468200,                 // B365 kickoff (may differ by minutes)
  "type": "INPLAY",
  "isTeamSwitch": false,
  "home_team_details": {
    "name_en": "Frayles de Guasave",
    "short_name_en": "Frayles de",
    "league": "Mexico CIBACOPA"
  },
  "away_team_details": {
    "name_en": "Rayos de Hermosillo",
    "short_name_en": "Rayos de H",
    "league": "Mexico CIBACOPA"
  },
  "our_event_id": "11471110",
  "entry_type": "MANUAL",                      // MANUAL = human mapped, AUTO = system
  "lastUpdatedBy": "Vasudha",                  // who did the mapping
  "sbo_mapped_id": "9479633",
  "flashscore_mapped_id": "QPF2xRgl",
  "r_id": "189929499C18A",
  "final_score": { "home": 111, "away": 114 }
}
```

### Decision Logic (How to Derive Labels)

```
IF mappedData EXISTS and mappedData.id is not null/empty:
    IF root.isTeamSwitch == true:
        decision = "SWAPPED"
    ELSE:
        decision = "MATCH"
ELSE (mappedData is null/absent/empty):
    decision = "NO_MATCH"
```

---

## 2. FIELD EXTRACTION MAP

Use these exact paths to extract data from the MongoDB documents:

| Field | JSON Path | Fallback |
|-------|-----------|----------|
| **OP match ID** | `doc.id` | `doc._id.$oid` |
| **OP home team** | `doc.home_team` | `doc.home_team_details.name_en` |
| **OP away team** | `doc.away_team` | `doc.away_team_details.name_en` |
| **OP league** | `doc.league.name` | `doc.league.league_name_en` |
| **OP sport** | `doc.sport` | — |
| **OP kickoff** | `doc.commence_time` | — (UNIX timestamp in seconds) |
| **OP category** | detect from league + team names | — |
| **B365 match ID** | `doc.mappedData.id` | `doc.bet365_mapped_id` |
| **B365 home team** | `doc.mappedData.home_team` | `doc.mappedData.home_team_details.name_en` |
| **B365 away team** | `doc.mappedData.away_team` | `doc.mappedData.away_team_details.name_en` |
| **B365 league** | `doc.mappedData.league.name` | `doc.mappedData.league.league_name_en` |
| **B365 kickoff** | `doc.mappedData.commence_time` | — (UNIX timestamp in seconds) |
| **Swap flag** | `doc.isTeamSwitch` | `false` |
| **Decision** | derived (see logic above) | — |
| **Mapped by** | `doc.mappedData.lastUpdatedBy` | `doc.lastUpdatedBy` |
| **Entry type** | `doc.mappedData.entry_type` | — (`MANUAL` or `AUTO`) |

---

## 3. PROJECT STRUCTURE

```
match-mapping-engine/
├── config/
│   ├── __init__.py
│   └── settings.py              # All thresholds + model paths (configurable)
├── core/
│   ├── __init__.py
│   ├── models.py                # Pydantic data models
│   ├── normalizer.py            # Team aliases, category detection, text cleaning
│   ├── inference.py             # SBERT retrieval + Cross-Encoder reranking
│   └── feedback.py              # Feedback ingestion + training pair generation
├── training/
│   ├── __init__.py
│   └── trainer.py               # SBERT + Cross-Encoder fine-tuning
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py             # Recall@K, Precision, drift monitoring
├── data/
│   ├── __init__.py
│   └── mongo_store.py           # MongoDB-backed production storage
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py          # ★ Convert MongoDB JSON → training data
│   ├── train_models.py          # Run model training
│   ├── evaluate.py              # Run offline evaluation
│   └── demo_pipeline.py         # End-to-end demo
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI inference API
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 4. FILE-BY-FILE BUILD INSTRUCTIONS

### 4.1 `requirements.txt`

```
torch>=2.0.0
sentence-transformers>=2.7.0
transformers>=4.40.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
numpy>=1.26.0
pandas>=2.2.0
pymongo>=4.7.0
scikit-learn>=1.5.0
tqdm>=4.66.0
```

### 4.2 `config/settings.py`

Centralized configuration. Every threshold must be configurable without code changes.

```python
from dataclasses import dataclass, field
import os

@dataclass
class ModelConfig:
    # Free HuggingFace models
    sbert_model: str = "all-MiniLM-L6-v2"                          # bi-encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reranker
    tuned_sbert_path: str = None          # after training
    tuned_cross_encoder_path: str = None  # after training
    use_tuned_sbert: bool = False         # feature flag
    use_tuned_cross_encoder: bool = False # feature flag
    sbert_top_k: int = 10                 # retrieval candidates
    rerank_top_k: int = 5                 # final output

@dataclass
class GateConfig:
    min_score: float = 0.90               # Score1 >= this
    margin: float = 0.10                  # Score1 - Score2 >= this
    kickoff_window_minutes: int = 30      # pre-filter window
    tight_kickoff_minutes: int = 15       # auto-match window
    sensitive_categories: list = field(default_factory=lambda: [
        "WOMEN", "U23", "U21", "U20", "U19", "U18", "U17",
        "RESERVES", "B-TEAM", "YOUTH", "AMATEUR"
    ])
    block_sensitive_auto_match: bool = True

@dataclass
class TrainingConfig:
    sbert_epochs: int = 3
    sbert_lr: float = 2e-5
    sbert_batch_size: int = 32
    ce_epochs: int = 3
    ce_lr: float = 2e-5
    ce_batch_size: int = 32
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    max_hard_negatives_per_positive: int = 4
```

### 4.3 `core/normalizer.py`

**Critical module.** This handles the messy real-world differences between OP and B365 team/league names.

Must implement:
- `clean_text(text)` → lowercase, strip accents, normalize whitespace
- `resolve_alias(team_name)` → map "Man Utd" → "manchester united" etc.
- `detect_categories(text)` → find [WOMEN], [U23], [RESERVES] etc. from league/team names
- `build_match_text(league, home, away, categories)` → `"[WOMEN] premier league | arsenal vs chelsea"`
- `build_swapped_text(...)` → same but home/away flipped

Category detection regex patterns (case-insensitive):
```
WOMEN:    \bwomen\b, \bwoman\b, \b\(w\)\b, \bladies\b, \bfemenino\b, \bfeminino\b, \bfrauen\b
U23:      \bu-?23\b, \bunder[\s-]?23\b, \bsub[\s-]?23\b
U21-U17:  same pattern with respective numbers
RESERVES: \breserves?\b, \bii\b, \b2nd\b
B-TEAM:   \bb[\s-]?team\b, \bcastilla\b
YOUTH:    \byouth\b, \bjunior\b, \bjuvenil\b
AMATEUR:  \bamateur\b
```

### 4.4 `scripts/prepare_data.py` — ★ MOST IMPORTANT FILE

This is the data pipeline that reads your MongoDB JSON exports and produces training data.

**Input**: JSON file containing array of MongoDB documents (the format shown in Section 1)

**Processing logic**:

```python
def load_and_convert(input_path: str) -> dict:
    """
    Load MongoDB JSON export → training pairs + labeled records.
    
    For EACH document in the array:
    
    1. EXTRACT OP fields:
       op_id        = doc["id"]
       op_home      = doc["home_team"]
       op_away      = doc["away_team"]
       op_league    = doc["league"]["name"] or doc["league"]["league_name_en"]
       op_sport     = doc["sport"]
       op_kickoff   = doc["commence_time"]  # UNIX timestamp (seconds)
    
    2. DETERMINE DECISION:
       if doc.get("mappedData") and doc["mappedData"].get("id"):
           if doc.get("isTeamSwitch", False) == True:
               decision = "SWAPPED"
           else:
               decision = "MATCH"
       else:
           decision = "NO_MATCH"
    
    3. EXTRACT B365 fields (only if MATCH or SWAPPED):
       mapped       = doc["mappedData"]
       b365_id      = mapped["id"]
       b365_home    = mapped["home_team"]
       b365_away    = mapped["away_team"]
       b365_league  = mapped["league"]["name"] or mapped["league"]["league_name_en"]
       b365_kickoff = mapped["commence_time"]  # UNIX timestamp (seconds)
    
    4. DETECT CATEGORIES from all text fields:
       categories = detect_categories(f"{op_league} {op_home} {op_away}")
    
    5. BUILD TEXT REPRESENTATIONS:
       op_text   = build_match_text(op_league, op_home, op_away, categories)
       b365_text = build_match_text(b365_league, b365_home, b365_away, b365_categories)
       # Example: "cibacopa | frayles de guasave vs rayos de hermosillo"
       # Example: "mexico cibacopa | frayles de guasave vs rayos de hermosillo"
    
    6. GENERATE TRAINING PAIRS:
       if decision == "MATCH":
           pairs.append(anchor=op_text, candidate=b365_text, label=1.0)
       elif decision == "SWAPPED":
           swapped_op = build_swapped_text(op_league, op_home, op_away, categories)
           pairs.append(anchor=swapped_op, candidate=b365_text, label=1.0)
       elif decision == "NO_MATCH":
           # No positive pair; store for hard negative mining later
           pass
    """
```

**Handle these edge cases**:
- `commence_time` is UNIX timestamp in **seconds** (not milliseconds)
- `league.name` may be `null` → fall back to `league.league_name_en`
- `mappedData` may be `null`, missing, or `{}`
- `isTeamSwitch` may be missing → default to `false`
- `_id.$oid` format from MongoDB exports → just use `doc["id"]` instead
- `$date` format for dates → parse with `datetime.fromisoformat()`
- Some team names have unicode characters → normalize with `unicodedata`
- League names differ between platforms: "Cibacopa" (OP) vs "Mexico CIBACOPA" (B365)

**Output files**:
1. `data/labeled_records.json` — standardized format for `train_models.py`
2. `data/training_pairs.json` — direct training pairs
3. `data/b365_pool.json` — all unique B365 matches (for indexing)
4. `data/stats.json` — conversion statistics

**CLI**:
```bash
python scripts/prepare_data.py --input raw_exports/matches.json --output-dir data/
python scripts/prepare_data.py --input raw_exports/ --output-dir data/  # directory of JSONs
```

### 4.5 `core/inference.py` — Inference Engine

Two-stage pipeline:

```
Pre-Filter (same sport + ±30min kickoff)
    ↓
SBERT Bi-Encoder: encode OP query + B365 pool → cosine similarity → Top-10
    ↓
Cross-Encoder Reranker: score (OP_text, B365_text) pairs → Top-5
    ↓
Swap Detection: also score (OP_swapped_text, B365_text), take best
    ↓
Gate Evaluation: AUTO_MATCH or NEED_REVIEW
    ↓
Output: MappingSuggestion JSON
```

**Key implementation details**:

1. **B365 Index**: Pre-encode all B365 matches with SBERT at startup. Store as numpy array.
2. **Pre-filter**: Filter by `sport` (exact match) and `commence_time` (±30 min window).
3. **SBERT retrieval**: Cosine similarity between OP query embedding and B365 embeddings. Return Top-10.
4. **Swap detection**: For each B365 candidate, score BOTH `(op_normal, b365)` AND `(op_swapped, b365)`. Keep the higher score and set `swapped=True` if swapped version won.
5. **Cross-encoder scores**: These are raw logits. Normalize to [0,1] with sigmoid: `1 / (1 + exp(-score))`.
6. **Always return Top-5**: Even if scores are low. Never return empty candidates.

### 4.6 `core/inference.py` — Gate Evaluation

```python
def evaluate_gates(op_match, candidates) -> (GateResult, details):
    """ALL four gates must pass for AUTO_MATCH."""
    
    top1 = candidates[0]
    top2_score = candidates[1].score if len(candidates) > 1 else 0.0
    
    # Gate 1: MinScore
    min_score_pass = top1.score >= CONFIG.gates.min_score  # default 0.90
    
    # Gate 2: Margin
    margin = top1.score - top2_score
    margin_pass = margin >= CONFIG.gates.margin  # default 0.10
    
    # Gate 3: Category
    # Sensitive categories (Women/U23/Reserves) CANNOT auto-match
    # Categories must match between OP and B365
    op_cats = set(op_match.category_tags)
    b365_cats = set(top1.category_tags)
    has_sensitive = bool((op_cats | b365_cats) & set(CONFIG.gates.sensitive_categories))
    category_pass = (not has_sensitive) and (op_cats == b365_cats)
    
    # Gate 4: Tight kickoff window
    kickoff_pass = top1.time_diff_minutes <= CONFIG.gates.tight_kickoff_minutes  # default 15
    
    if all([min_score_pass, margin_pass, category_pass, kickoff_pass]):
        return "AUTO_MATCH", details
    else:
        return "NEED_REVIEW", details
```

### 4.7 `core/feedback.py` — Feedback Ingestion

Consumes human decisions and generates training pairs.

**Rules**:
- MATCH → `(op_text, selected_b365_text)` = positive (label=1.0)
- SWAPPED → `(op_swapped_text, selected_b365_text)` = positive (label=1.0)
- Unselected Top-5 candidates → hard negatives (label=0.0)
- NO_MATCH → ALL candidates = hard negatives (label=0.0)
- **NEVER** include the true positive in negative pool (label contamination)
- Enforce **1:1 mapping**: one OP match → one B365 match, one B365 match → one OP match

### 4.8 `training/trainer.py` — Model Training

**SBERT Training**:
- Loss: `MultipleNegativesRankingLoss` (uses in-batch negatives automatically)
- Input: only POSITIVE pairs `(op_text, b365_text)`
- The loss function treats other pairs in the same batch as negatives

**Cross-Encoder Training**:
- Loss: Binary classification (BCEWithLogitsLoss)
- Input: both positives (label=1.0) and hard negatives (label=0.0)
- Hard negatives = unselected Top-5 candidates from real suggestions

**Data split**: Group by suggestion_id to prevent leakage between train/val/test.

**Contamination check**: Before every training run, verify no (anchor, candidate) pair appears as both positive AND negative.

### 4.9 `evaluation/evaluator.py` — Offline Evaluation

Metrics to compute:

| Metric | Formula | Target |
|--------|---------|--------|
| **Recall@5** | (matches where true B365 is in Top-5) / (total match-exists cases) | ≥ 90% |
| **Recall@10** | (matches where true B365 is in Top-10) / (total match-exists cases) | ≥ 95% |
| **Precision@1** | (Top-1 is correct) / (total match-exists cases) | — |
| **AUTO_MATCH Precision** | (correct auto-matches) / (total auto-matches) | ≥ 98% |
| **AUTO_MATCH Rate** | (auto-matches) / (total predictions) | increasing |
| **No-Match FP Rate** | (auto-matched a NO_MATCH case) / (total NO_MATCH cases) | < 2% |

Also compute per-sport and per-category breakdowns for drift monitoring.

### 4.10 `api/server.py` — FastAPI

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Top-5 candidates + gate decision for one OP match |
| POST | `/predict/batch` | Batch predictions |
| POST | `/index/refresh` | Upload + index B365 matches |
| POST | `/feedback` | Submit human decision |
| POST | `/models/reload` | Hot-swap between base and tuned models |
| GET | `/health` | Health check |
| GET | `/metrics` | Feedback + training pair counts |

---

## 5. DATA PREPARATION SCRIPT — DETAILED SPEC

This is the script the user will run first. It reads their MongoDB JSON exports.

### Input Format

The user will provide JSON files. Each file is either:
- A JSON array of documents: `[ { doc1 }, { doc2 }, ... ]`
- A single document: `{ ... }`
- A mongoexport with `$oid` / `$date` wrappers

### Parsing Rules

```python
def parse_document(doc: dict) -> dict:
    """Parse a single MongoDB document into standardized format."""
    
    # ── OP Fields ──
    op_id = doc.get("id") or str(doc.get("_id", {}).get("$oid", ""))
    op_home = doc.get("home_team", "")
    op_away = doc.get("away_team", "")
    op_sport = doc.get("sport", "")
    
    # League: try multiple paths
    league_obj = doc.get("league", {})
    op_league = (
        league_obj.get("name")
        or league_obj.get("league_name_en")
        or ""
    )
    
    # Kickoff: UNIX timestamp in seconds
    op_kickoff = doc.get("commence_time")
    if op_kickoff:
        op_kickoff_dt = datetime.utcfromtimestamp(int(op_kickoff))
    
    # ── Decision ──
    mapped_data = doc.get("mappedData")
    has_mapping = (
        mapped_data is not None
        and isinstance(mapped_data, dict)
        and mapped_data.get("id")  # must have a B365 ID
    )
    
    if has_mapping:
        is_swapped = doc.get("isTeamSwitch", False)
        decision = "SWAPPED" if is_swapped else "MATCH"
    else:
        decision = "NO_MATCH"
    
    # ── B365 Fields (from mappedData) ──
    b365_id = ""
    b365_home = ""
    b365_away = ""
    b365_league = ""
    b365_kickoff = None
    
    if has_mapping:
        b365_id = mapped_data.get("id", "")
        b365_home = mapped_data.get("home_team", "")
        b365_away = mapped_data.get("away_team", "")
        
        b365_league_obj = mapped_data.get("league", {})
        b365_league = (
            b365_league_obj.get("name")
            or b365_league_obj.get("league_name_en")
            or ""
        )
        
        b365_kickoff = mapped_data.get("commence_time")
    
    return {
        "op_match_id": op_id,
        "op_home": op_home,
        "op_away": op_away,
        "op_league": op_league,
        "op_sport": op_sport,
        "op_kickoff": op_kickoff,
        "b365_match_id": b365_id if has_mapping else None,
        "b365_home": b365_home,
        "b365_away": b365_away,
        "b365_league": b365_league,
        "b365_kickoff": b365_kickoff,
        "decision": decision,
        "is_swapped": doc.get("isTeamSwitch", False),
        "mapped_by": mapped_data.get("lastUpdatedBy", "") if mapped_data else "",
        "entry_type": mapped_data.get("entry_type", "") if mapped_data else "",
    }
```

### Generate Training Pairs

```python
def generate_training_pairs(parsed_records: list) -> list:
    pairs = []
    
    for rec in parsed_records:
        # Detect categories
        op_cats = detect_categories(f"{rec['op_league']} {rec['op_home']} {rec['op_away']}")
        b365_cats = detect_categories(f"{rec['b365_league']} {rec['b365_home']} {rec['b365_away']}")
        
        # Build text
        op_text = build_match_text(rec["op_league"], rec["op_home"], rec["op_away"], op_cats)
        b365_text = build_match_text(rec["b365_league"], rec["b365_home"], rec["b365_away"], b365_cats)
        
        if rec["decision"] == "MATCH":
            pairs.append({
                "anchor_text": op_text,
                "candidate_text": b365_text,
                "label": 1.0,
                "is_hard_negative": False
            })
        
        elif rec["decision"] == "SWAPPED":
            swapped_op = build_swapped_text(
                rec["op_league"], rec["op_home"], rec["op_away"], op_cats
            )
            pairs.append({
                "anchor_text": swapped_op,
                "candidate_text": b365_text,
                "label": 1.0,
                "is_hard_negative": False
            })
        
        elif rec["decision"] == "NO_MATCH":
            # If there was a B365 candidate shown, it's a hard negative
            if rec["b365_home"] and rec["b365_away"]:
                pairs.append({
                    "anchor_text": op_text,
                    "candidate_text": b365_text,
                    "label": 0.0,
                    "is_hard_negative": True
                })
    
    return pairs
```

### Also Build B365 Pool

Extract ALL unique B365 matches for the inference engine's index:

```python
def extract_b365_pool(parsed_records: list) -> list:
    """Extract unique B365 matches for indexing."""
    seen = set()
    pool = []
    
    for rec in parsed_records:
        b365_id = rec.get("b365_match_id")
        if not b365_id or b365_id in seen:
            continue
        seen.add(b365_id)
        
        pool.append({
            "match_id": b365_id,
            "platform": "B365",
            "sport": rec["op_sport"],
            "league": rec["b365_league"],
            "home_team": rec["b365_home"],
            "away_team": rec["b365_away"],
            "kickoff": rec["b365_kickoff"],
        })
    
    return pool
```

---

## 6. TRAINING PIPELINE — STEP BY STEP

### Step 1: Prepare Data
```bash
python scripts/prepare_data.py --input /path/to/mongodb_export.json --output-dir data/
```

Output:
- `data/labeled_records.json` — for training
- `data/training_pairs.json` — direct pairs
- `data/b365_pool.json` — for B365 index
- `data/stats.json` — summary

### Step 2: Train Models
```bash
python scripts/train_models.py --data-file data/labeled_records.json
```

Output:
- `models/sbert_tuned_YYYYMMDD/` — tuned SBERT
- `models/ce_tuned_YYYYMMDD/` — tuned Cross-Encoder

### Step 3: Evaluate
```bash
python scripts/evaluate.py \
  --test-data data/labeled_records.json \
  --sbert-model models/sbert_tuned_YYYYMMDD \
  --ce-model models/ce_tuned_YYYYMMDD
```

### Step 4: Deploy
```bash
# Start API
uvicorn api.server:app --port 8000

# Load B365 pool
curl -X POST http://localhost:8000/index/refresh \
  -d @data/b365_pool.json

# Switch to tuned models
curl -X POST http://localhost:8000/models/reload \
  -d '{"use_tuned_sbert": true, "tuned_sbert_path": "models/sbert_tuned_YYYYMMDD", ...}'

# Predict
curl -X POST http://localhost:8000/predict -d '{"op_match": {...}}'
```

---

## 7. HARD NEGATIVE MINING (Active Learning)

After the model is deployed and humans review suggestions:

1. Human picks B365 match #3 from Top-5 → 
   - #3 becomes positive pair
   - #1, #2, #4, #5 become **hard negatives** (model ranked them high but they're wrong)

2. Human clicks NO_MATCH →
   - ALL Top-5 become hard negatives

3. These hard negatives are the most valuable training signal because they represent the model's current failure modes.

4. Periodically retrain with accumulated feedback → model improves → automation rate increases.

---

## 8. IMPORTANT EDGE CASES TO HANDLE

1. **League name differences**: OP has "Cibacopa", B365 has "Mexico CIBACOPA" → text normalization must handle this.

2. **Kickoff time differences**: OP `commence_time: 1771469719` vs B365 `commence_time: 1771468200` = ~25 min difference. Pre-filter must use ±30 min window.

3. **`isTeamSwitch` flag**: When `true`, the OP home/away are swapped relative to B365. The model must learn this.

4. **Missing `mappedData`**: NO_MATCH case. No B365 match exists in the pool.

5. **`$oid` and `$date` wrappers**: MongoDB export format. Parse `{"$oid": "..."}` → just the string, `{"$date": "..."}` → datetime.

6. **Multiple platforms**: Documents may have `sbo_mapped_id`, `flashscore_mapped_id` in addition to B365. Focus only on B365 (`mappedData` where `platform == "BET365"`).

7. **`entry_type: "MANUAL"` vs `"AUTO"`**: Track this. Manual entries are higher quality labels. Auto entries may have been from a previous system and could be noisy.

8. **Sport ID mapping**: `sport_id: "4"` = Basketball. Use `sport` string field instead, it's more reliable.

---

## 9. SUCCESS CRITERIA

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Recall@5 | ≥ 90% | Correct B365 appears in Top-5 candidates |
| AUTO_MATCH Precision | ≥ 98% | When system auto-confirms, it's correct |
| No-Match FP Rate | < 2% | System doesn't auto-match when no match exists |
| Automation Rate | Increasing weekly | More AUTO_MATCH, fewer NEED_REVIEW |

---

## 10. MODELS USED (ALL FREE)

| Model | HuggingFace ID | Purpose | Size |
|-------|---------------|---------|------|
| SBERT Bi-Encoder | `all-MiniLM-L6-v2` | Fast retrieval (cosine similarity) | 80MB |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise reranking | 80MB |

Both are Apache 2.0 licensed, free for commercial use, and fine-tunable with `sentence-transformers` library.

After fine-tuning on your ~10,000 back office mappings, these models will learn your specific domain patterns (team aliases, league name variations, sport-specific quirks).
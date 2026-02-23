# Training Data Formats

This document describes all data formats the AI Match Mapping Engine accepts for training.

There are **3 ways** to feed training data into the system:

| Method | Input Format | Script | When to Use |
|---|---|---|---|
| **1. Labeled Records JSON** | Flat JSON array | `scripts/train_models.py` | You have pre-labeled match pairs |
| **2. MongoDB JSON Exports** | Raw MongoDB export | `scripts/prepare_data.py` | You have MongoDB database dumps |
| **3. CSE Feedback API** | Fetched automatically | `scripts/self_train_pipeline.py` | Live feedback from CSE team (auto) |

---

## 1. Labeled Records JSON (Primary Format)

This is the main format used by `scripts/train_models.py`. Each record represents one provider match paired with a Bet365 match and a human decision.

### File: `data/labeled_records.json`

```json
[
  {
    "op_match_id": "lnKoS5iC",
    "op_home": "Loughgall",
    "op_away": "Knockbreda",
    "op_league": "Irish Cup",
    "op_sport": "Soccer",
    "op_kickoff": 1770147900,
    "b365_match_id": "188358852",
    "b365_home": "Loughgall",
    "b365_away": "Knockbreda",
    "b365_league": "Irish Cup",
    "b365_kickoff": 1768938300,
    "decision": "MATCH",
    "is_swapped": false
  },
  {
    "op_match_id": "Gtb8aME4",
    "op_home": "Bronshoj (Den)",
    "op_away": "Helsingor (Den)",
    "op_league": "Europe Friendlies",
    "op_sport": "Soccer",
    "op_kickoff": 1769700600,
    "b365_match_id": "188773137",
    "b365_home": "Bronshoj BK",
    "b365_away": "FC Helsingor",
    "b365_league": "Europe Friendlies",
    "b365_kickoff": 1769700600,
    "decision": "SWAPPED",
    "is_swapped": true
  },
  {
    "op_match_id": "xyz123",
    "op_home": "Team A",
    "op_away": "Team B",
    "op_league": "Some League",
    "op_sport": "Soccer",
    "op_kickoff": 1770000000,
    "b365_match_id": null,
    "b365_home": "",
    "b365_away": "",
    "b365_league": "",
    "b365_kickoff": null,
    "decision": "NO_MATCH",
    "is_swapped": false
  }
]
```

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `op_match_id` | string | Yes | Provider match ID (OddsPortal, FlashScore, etc.) |
| `op_home` | string | Yes | Provider home team name |
| `op_away` | string | Yes | Provider away team name |
| `op_league` | string | Yes | Provider league name |
| `op_sport` | string | Yes | Sport name (e.g., `Soccer`, `Basketball`, `Tennis`, `Volleyball`) |
| `op_kickoff` | integer | Yes | Kickoff time as UNIX timestamp (seconds) |
| `b365_match_id` | string/null | Yes | Bet365 match ID (null for `NO_MATCH`) |
| `b365_home` | string | Yes | Bet365 home team name (empty for `NO_MATCH`) |
| `b365_away` | string | Yes | Bet365 away team name (empty for `NO_MATCH`) |
| `b365_league` | string | Yes | Bet365 league name (empty for `NO_MATCH`) |
| `b365_kickoff` | integer/null | No | Bet365 kickoff UNIX timestamp |
| `decision` | string | Yes | One of: `MATCH`, `SWAPPED`, `NO_MATCH` |
| `is_swapped` | boolean | No | `true` if home/away teams are swapped between platforms |

### Decision Values

| Decision | Meaning | Training Effect |
|---|---|---|
| `MATCH` | The Bet365 match is the correct mapping | **Positive pair** — model learns to score this higher |
| `SWAPPED` | Correct match but home/away teams are reversed | **Positive pair (swapped)** — model learns correct team ordering |
| `NO_MATCH` | No correct Bet365 match exists for this provider match | **Hard negative** — model learns to score this lower |

### Run Training

```bash
python scripts/train_models.py --data-file data/labeled_records.json
```

---

## 2. MongoDB JSON Exports

Raw MongoDB exports from your database. Place these files in the `training_data/` folder.

### File: `training_data/*.json`

The script `prepare_data.py` parses the MongoDB document structure and converts it into the Labeled Records format (Format 1).

Each document should look like:

```json
{
  "_id": { "$oid": "697990e4d11696bcc660f33a" },
  "home_team": "Mount Pleasant",
  "away_team": "Arnett Gardens",
  "league": {
    "name": "Jamaica Premier League"
  },
  "sport_id": "1",
  "commence_time": 1769716800,
  "isTeamSwitch": false,
  "mappedData": {
    "id": "188456123",
    "home_team": "Mount Pleasant Academy",
    "away_team": "Arnett Gardens FC",
    "league": {
      "name": "Jamaica Premier League"
    },
    "commence_time": 1769716800,
    "entry_type": "manual"
  }
}
```

### Key Fields

| Field | Type | Description |
|---|---|---|
| `_id` | object/string | MongoDB document ID |
| `home_team` | string | Provider home team |
| `away_team` | string | Provider away team |
| `league` | object | `{ "name": "League Name" }` |
| `sport_id` | string | Sport ID (`1`=Soccer, `2`=Tennis, `3`=Basketball, etc.) |
| `commence_time` | integer | Kickoff UNIX timestamp |
| `isTeamSwitch` | boolean | If `true`, decision becomes `SWAPPED` |
| `mappedData` | object/null | The matched Bet365 record. If `null` or missing `id`, decision is `NO_MATCH` |
| `mappedData.id` | string | Bet365 match ID |
| `mappedData.home_team` | string | Bet365 home team |
| `mappedData.away_team` | string | Bet365 away team |
| `mappedData.league.name` | string | Bet365 league name |
| `mappedData.commence_time` | integer | Bet365 kickoff UNIX timestamp |

### How It Works

- Document has `mappedData` with valid `id` → **MATCH** (or **SWAPPED** if `isTeamSwitch=true`)
- Document has no `mappedData` or `mappedData.id` is empty → **NO_MATCH**

### Run Data Preparation + Training

```bash
# Step 1: Place your MongoDB JSON files in training_data/
# Step 2: Convert to labeled records
python scripts/prepare_data.py

# Step 3: Train models on the converted data
python scripts/train_models.py --data-file data/labeled_records.json
```

You can place **multiple JSON files** in the `training_data/` folder — all will be processed.

---

## 3. CSE Feedback API v2 (Automatic)

The self-training pipeline fetches feedback automatically from the CSE team API v2 endpoint. No manual data preparation needed.

**API Endpoint:** `/matches/get-ai-mapping-feedback-v2`

### API Response Format

Each feedback record from the v2 API looks like this. The provider match data is inside `provider_data[]` and the Bet365 match data is inside `bet365_match[]`:

```json
{
  "_id": { "$oid": "699841b5d11696bcc6b7666f" },
  "platform": "ODDSPORTAL",
  "provider_id": "h0hmpk5F",
  "confidence": 0,
  "is_mapped": true,
  "is_checked": true,
  "reason": "ai_moderate_confidence",
  "switch": false,
  "feedback": "Not Correct",
  "logs": [
    {
      "what": "Not Correct",
      "who": "Sriram",
      "when": { "$date": "2026-02-20T11:25:38.047Z" }
    }
  ],
  "bet365_match": [
    {
      "id": "189528039",
      "home_team": "Gembo Borgerhout",
      "away_team": "Basket SKT Ieper",
      "sport": "Basketball",
      "sport_id": "4",
      "commence_time": 1771702200,
      "league": {
        "name": "Belgium Top Division 1"
      },
      "entry_type": "MANUAL"
    }
  ],
  "provider_data": [
    {
      "id": "h0hmpk5F",
      "home_team": "Houston",
      "away_team": "Arizona",
      "sport": "Basketball",
      "sport_id": "4",
      "commence_time": 1771703354,
      "platform": "ODDSPORTAL",
      "league": {
        "name": "NCAA Division I Men's Basketball"
      }
    }
  ]
}
```

### Key Fields Extracted for Training

| Field | Source | Description |
|---|---|---|
| `feedback` | Root level | CSE reviewer's decision |
| `provider_id` | Root level | Provider match ID |
| `platform` | Root level | Provider platform name |
| `provider_data[0].home_team` | Nested array | Provider home team name |
| `provider_data[0].away_team` | Nested array | Provider away team name |
| `provider_data[0].league.name` | Nested object | Provider league name |
| `provider_data[0].sport` | Nested object | Sport name |
| `bet365_match[0].home_team` | Nested array | Bet365 home team name |
| `bet365_match[0].away_team` | Nested array | Bet365 away team name |
| `bet365_match[0].league.name` | Nested object | Bet365 league name |

### CSE Feedback Values

| Feedback | Internal Decision | Training Effect |
|---|---|---|
| `Correct` | MATCH | Positive pair — boosts match score |
| `Not Correct` | NO_MATCH | Hard negative — lowers match score |
| `Need to swap` | SWAPPED | Positive pair with swapped teams |
| `Not Sure` | *(skipped)* | No training signal — uncertain data is ignored |

### Run Self-Training

```bash
# Fetch feedback and train (automatic)
python scripts/self_train_pipeline.py --platform ODDSPORTAL

# Or via the scheduler (runs everything)
python scripts/scheduler.py --run-once
```

---

## 4. Sport ID Mapping

When `sport_id` is provided instead of a sport name, this mapping is used:

| sport_id | Sport |
|---|---|
| 1 | Soccer |
| 2 | Tennis |
| 3 | Basketball |
| 4 | Hockey |
| 5 | Volleyball |
| 6 | Handball |
| 7 | Baseball |
| 8 | American Football |
| 9 | Rugby |
| 10 | Boxing |
| 12 | Table Tennis |
| 13 | Cricket |
| 18 | Esports |

---

## 5. Output Format (AI Results)

After inference, each result pushed to the API or saved to file looks like:

```json
{
  "platform": "ODDSPORTAL",
  "bet365_match": "188358852",
  "provider_id": "lnKoS5iC",
  "confidence": 0.97,
  "is_checked": false,
  "is_mapped": true,
  "reason": "rule_based_strong_match",
  "switch": false
}
```

| Field | Type | Description |
|---|---|---|
| `platform` | string | Provider name (`ODDSPORTAL`, `FLASHSCORE`, `SOFASCORE`, `SBO`) |
| `bet365_match` | string/null | Matched Bet365 ID (null entries are never pushed) |
| `provider_id` | string | Provider's match ID |
| `confidence` | float | AI confidence score (0.0 - 1.0) |
| `is_checked` | boolean | Always `false` (set to `true` after human review) |
| `is_mapped` | boolean | `true` if confidence >= threshold and bet365_match exists |
| `reason` | string | `rule_based_strong_match`, `ai_high_confidence`, `ai_moderate_confidence`, `ai_low_confidence`, `no_candidates` |
| `switch` | boolean | `true` if home/away are swapped between platforms |

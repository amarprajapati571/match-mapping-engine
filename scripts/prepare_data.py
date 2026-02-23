"""
Data Preparation — Convert MongoDB JSON exports into training data.

Reads JSON files from training_data/ folder, parses the MongoDB document
structure (OddsPortal matches with embedded mappedData for Bet365), and
produces labeled records + training pairs for model fine-tuning.

Input:  MongoDB JSON exports placed in training_data/ folder
Output: data/labeled_records.json, data/training_pairs.json,
        data/b365_pool.json, data/prepare_stats.json

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input training_data/
    python scripts/prepare_data.py --input training_data/my_export.json
    python scripts/prepare_data.py --input training_data/ --output-dir data/
"""

import sys
import os
import json
import glob
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.normalizer import (
    build_match_text,
    build_swapped_text,
    detect_categories,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("prepare_data")


# ═══════════════════════════════════════════════════════════════════
# MongoDB Document Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_document(doc: dict) -> Optional[dict]:
    """
    Parse a single MongoDB document (OddsPortal match with optional
    mappedData for Bet365) into a standardized labeled record.

    Returns None if the document is unparseable or missing required fields.
    """
    try:
        # ── OP Fields ──
        op_id = doc.get("id") or _extract_oid(doc.get("_id"))
        if not op_id:
            return None

        op_home = (
            doc.get("home_team", "")
            or _nested(doc, "home_team_details", "name_en", "")
        )
        op_away = (
            doc.get("away_team", "")
            or _nested(doc, "away_team_details", "name_en", "")
        )
        if not op_home or not op_away:
            return None

        league_obj = doc.get("league") or {}
        op_league = (
            league_obj.get("name")
            or league_obj.get("league_name_en")
            or ""
        )
        op_sport = doc.get("sport", "")
        op_kickoff = _parse_timestamp(doc.get("commence_time"))

        # ── Decision ──
        mapped_data = doc.get("mappedData")
        has_mapping = (
            mapped_data is not None
            and isinstance(mapped_data, dict)
            and bool(mapped_data.get("id"))
        )

        if has_mapping:
            is_swapped = doc.get("isTeamSwitch", False) is True
            decision = "SWAPPED" if is_swapped else "MATCH"
        else:
            decision = "NO_MATCH"

        # ── B365 Fields ──
        b365_id = ""
        b365_home = ""
        b365_away = ""
        b365_league = ""
        b365_kickoff = None

        if has_mapping:
            b365_id = str(mapped_data.get("id", ""))
            b365_home = (
                mapped_data.get("home_team", "")
                or _nested(mapped_data, "home_team_details", "name_en", "")
            )
            b365_away = (
                mapped_data.get("away_team", "")
                or _nested(mapped_data, "away_team_details", "name_en", "")
            )
            b365_league_obj = mapped_data.get("league") or {}
            b365_league = (
                b365_league_obj.get("name")
                or b365_league_obj.get("league_name_en")
                or ""
            )
            b365_kickoff = _parse_timestamp(mapped_data.get("commence_time"))

        return {
            "op_match_id": str(op_id),
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
            "is_swapped": doc.get("isTeamSwitch", False) is True,
            "mapped_by": (mapped_data.get("lastUpdatedBy", "") if mapped_data else ""),
            "entry_type": (mapped_data.get("entry_type", "") if mapped_data else ""),
        }
    except Exception as e:
        logger.warning(f"Failed to parse document: {e}")
        return None


def _extract_oid(value) -> str:
    """Extract string ID from MongoDB $oid wrapper or plain value."""
    if isinstance(value, dict):
        return str(value.get("$oid", ""))
    return str(value) if value else ""


def _nested(d: dict, key1: str, key2: str, default=""):
    """Safely get nested dict value."""
    sub = d.get(key1)
    if isinstance(sub, dict):
        return sub.get(key2, default)
    return default


def _parse_timestamp(value) -> Optional[int]:
    """Parse various timestamp formats to UNIX seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1e12:
            ts = ts // 1000
        return ts
    if isinstance(value, dict) and "$date" in value:
        try:
            dt = datetime.fromisoformat(value["$date"].replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return None
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════
# Training Pair Generation
# ═══════════════════════════════════════════════════════════════════

def generate_training_pairs(parsed_records: List[dict]) -> List[dict]:
    """Convert parsed labeled records into training pairs."""
    pairs = []

    for rec in parsed_records:
        decision = rec["decision"]
        if decision == "NO_MATCH":
            if rec.get("b365_home") and rec.get("b365_away"):
                op_cats = detect_categories(
                    f"{rec['op_league']} {rec['op_home']} {rec['op_away']}"
                )
                b365_cats = detect_categories(
                    f"{rec['b365_league']} {rec['b365_home']} {rec['b365_away']}"
                )
                op_text = build_match_text(
                    rec["op_league"], rec["op_home"], rec["op_away"], op_cats
                )
                b365_text = build_match_text(
                    rec["b365_league"], rec["b365_home"], rec["b365_away"], b365_cats
                )
                pairs.append({
                    "anchor_text": op_text,
                    "candidate_text": b365_text,
                    "label": 0.0,
                    "is_hard_negative": True,
                    "source_suggestion_id": f"old_data_{rec['op_match_id']}",
                })
            continue

        op_cats = detect_categories(
            f"{rec['op_league']} {rec['op_home']} {rec['op_away']}"
        )
        b365_cats = detect_categories(
            f"{rec['b365_league']} {rec['b365_home']} {rec['b365_away']}"
        )
        b365_text = build_match_text(
            rec["b365_league"], rec["b365_home"], rec["b365_away"], b365_cats
        )

        if decision == "MATCH":
            op_text = build_match_text(
                rec["op_league"], rec["op_home"], rec["op_away"], op_cats
            )
            pairs.append({
                "anchor_text": op_text,
                "candidate_text": b365_text,
                "label": 1.0,
                "is_hard_negative": False,
                "source_suggestion_id": f"old_data_{rec['op_match_id']}",
            })

        elif decision == "SWAPPED":
            swapped_text = build_swapped_text(
                rec["op_league"], rec["op_home"], rec["op_away"], op_cats
            )
            pairs.append({
                "anchor_text": swapped_text,
                "candidate_text": b365_text,
                "label": 1.0,
                "is_hard_negative": False,
                "source_suggestion_id": f"old_data_{rec['op_match_id']}",
            })
            op_text = build_match_text(
                rec["op_league"], rec["op_home"], rec["op_away"], op_cats
            )
            pairs.append({
                "anchor_text": op_text,
                "candidate_text": b365_text,
                "label": 0.0,
                "is_hard_negative": True,
                "source_suggestion_id": f"old_data_{rec['op_match_id']}",
            })

    return pairs


def extract_b365_pool(parsed_records: List[dict]) -> List[dict]:
    """Extract unique B365 matches for the inference engine index."""
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


# ═══════════════════════════════════════════════════════════════════
# File Loading
# ═══════════════════════════════════════════════════════════════════

def load_json_files(input_path: str) -> List[dict]:
    """
    Load MongoDB JSON exports from a file or directory.

    Handles:
      - Single JSON file (array of documents or single document)
      - Directory of JSON files
      - MongoDB export with $oid/$date wrappers
    """
    files: List[str] = []

    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = sorted(
            glob.glob(os.path.join(input_path, "*.json"))
            + glob.glob(os.path.join(input_path, "**/*.json"), recursive=True)
        )
        files = list(dict.fromkeys(files))
    else:
        logger.error(f"Input path not found: {input_path}")
        return []

    if not files:
        logger.error(f"No JSON files found in {input_path}")
        return []

    all_docs: List[dict] = []

    for fpath in files:
        logger.info(f"Loading {fpath}...")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                all_docs.extend(data)
                logger.info(f"  → {len(data)} documents")
            elif isinstance(data, dict):
                all_docs.append(data)
                logger.info(f"  → 1 document")
            else:
                logger.warning(f"  → Unexpected format, skipping")
        except json.JSONDecodeError as e:
            logger.warning(f"  → JSON parse error: {e}")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                            if isinstance(doc, dict):
                                all_docs.append(doc)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"  → Loaded as JSONL (line-delimited)")
            except Exception:
                logger.warning(f"  → Skipping unreadable file")
        except Exception as e:
            logger.warning(f"  → Error: {e}")

    logger.info(f"Total raw documents loaded: {len(all_docs)}")
    return all_docs


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert MongoDB JSON exports → training data for match mapping models"
    )
    parser.add_argument(
        "--input", default="training_data/",
        help="Path to JSON file or directory of JSON files (default: training_data/)",
    )
    parser.add_argument(
        "--output-dir", default="data/",
        help="Output directory for processed data (default: data/)",
    )
    parser.add_argument(
        "--manual-only", action="store_true",
        help="Only include MANUAL entry_type records (higher quality labels)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  DATA PREPARATION — MongoDB JSON → Training Data")
    print("=" * 70 + "\n")

    # ── Step 1: Load JSON files ──
    print(f"Step 1: Loading JSON files from {args.input}...")
    raw_docs = load_json_files(args.input)

    if not raw_docs:
        print("\n  No documents found. Place JSON files in training_data/ folder.")
        print("  See GUIDE.md for expected format.\n")
        return

    print(f"  Loaded {len(raw_docs)} raw documents\n")

    # ── Step 2: Parse documents ──
    print("Step 2: Parsing MongoDB documents...")
    parsed_records = []
    parse_errors = 0

    for doc in raw_docs:
        rec = parse_document(doc)
        if rec is None:
            parse_errors += 1
            continue
        if args.manual_only and rec.get("entry_type") != "MANUAL":
            continue
        parsed_records.append(rec)

    decision_counts = {}
    sport_counts = {}
    for rec in parsed_records:
        d = rec["decision"]
        s = rec["op_sport"]
        decision_counts[d] = decision_counts.get(d, 0) + 1
        sport_counts[s] = sport_counts.get(s, 0) + 1

    print(f"  Parsed: {len(parsed_records)} records ({parse_errors} errors)")
    print(f"  Decisions: {json.dumps(decision_counts)}")
    print(f"  Sports:    {json.dumps(sport_counts)}\n")

    if not parsed_records:
        print("  No valid records after parsing. Check your JSON format.")
        return

    # ── Step 3: Generate training pairs ──
    print("Step 3: Generating training pairs...")
    training_pairs = generate_training_pairs(parsed_records)

    positives = sum(1 for p in training_pairs if p["label"] == 1.0)
    negatives = sum(1 for p in training_pairs if p["label"] == 0.0)

    print(f"  Training pairs: {len(training_pairs)}")
    print(f"    Positives:      {positives}")
    print(f"    Hard negatives: {negatives}\n")

    # ── Step 4: Extract B365 pool ──
    print("Step 4: Extracting B365 match pool...")
    b365_pool = extract_b365_pool(parsed_records)
    print(f"  Unique B365 matches: {len(b365_pool)}\n")

    # ── Step 5: Save outputs ──
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Step 5: Saving to {args.output_dir}...")

    labeled_path = os.path.join(args.output_dir, "labeled_records.json")
    pairs_path = os.path.join(args.output_dir, "training_pairs.json")
    pool_path = os.path.join(args.output_dir, "b365_pool.json")
    stats_path = os.path.join(args.output_dir, "prepare_stats.json")

    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2, ensure_ascii=False, default=str)

    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False, default=str)

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(b365_pool, f, indent=2, ensure_ascii=False, default=str)

    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": args.input,
        "raw_documents": len(raw_docs),
        "parse_errors": parse_errors,
        "parsed_records": len(parsed_records),
        "decisions": decision_counts,
        "sports": sport_counts,
        "training_pairs": len(training_pairs),
        "positives": positives,
        "hard_negatives": negatives,
        "b365_pool_size": len(b365_pool),
        "manual_only": args.manual_only,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    file_sizes = {
        labeled_path: os.path.getsize(labeled_path),
        pairs_path: os.path.getsize(pairs_path),
        pool_path: os.path.getsize(pool_path),
        stats_path: os.path.getsize(stats_path),
    }

    print(f"  {labeled_path} ({file_sizes[labeled_path] / 1024:.1f} KB)")
    print(f"  {pairs_path} ({file_sizes[pairs_path] / 1024:.1f} KB)")
    print(f"  {pool_path} ({file_sizes[pool_path] / 1024:.1f} KB)")
    print(f"  {stats_path} ({file_sizes[stats_path] / 1024:.1f} KB)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"""
  Input:
    Source:           {args.input}
    Raw documents:    {len(raw_docs)}
    Parsed records:   {len(parsed_records)}

  Decisions:
    MATCH:     {decision_counts.get('MATCH', 0)}
    SWAPPED:   {decision_counts.get('SWAPPED', 0)}
    NO_MATCH:  {decision_counts.get('NO_MATCH', 0)}

  Training Data:
    Total pairs:      {len(training_pairs)}
    Positives:        {positives}
    Hard negatives:   {negatives}
    B365 pool:        {len(b365_pool)} unique matches

  Output Files:
    {labeled_path}
    {pairs_path}
    {pool_path}
    {stats_path}

  Next Steps:
    1. Train models:
       python scripts/train_models.py --data-file {labeled_path}

    2. Or train with custom params:
       python scripts/train_models.py --data-file {labeled_path} --epochs 5 --batch-size 16
""")


if __name__ == "__main__":
    main()

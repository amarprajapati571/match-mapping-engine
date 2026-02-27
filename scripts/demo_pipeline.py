"""
End-to-End Demo Script.

Demonstrates the complete pipeline:
1. Generate synthetic data (simulating OP + B365 feeds)
2. Index B365 pool
3. Run inference → Top-5 candidates + gate decisions
4. Ingest human feedback → training pairs
5. Build datasets + train models
6. Run offline evaluation → report
7. Model hot-swap via feature flag

Run: python scripts/demo_pipeline.py
"""

import sys
import os
import json
import random
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.models import MatchRecord, Decision, FeedbackRecord
from core.inference import InferenceEngine
from core.feedback import FeedbackStore, BulkFeedbackLoader
from core.normalizer import detect_categories, build_match_text
from training.trainer import TrainingOrchestrator, DatasetBuilder
from evaluation.evaluator import Evaluator, EvalDataPoint, ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("demo")


# ═══════════════════════════════════════════════
# Step 0: Generate Synthetic Data
# ═══════════════════════════════════════════════

def generate_synthetic_data(n_matches: int = 200):
    """Generate realistic OP + B365 match pairs for demo."""
    
    leagues = [
        ("soccer", "Premier League"),
        ("soccer", "La Liga"),
        ("soccer", "Bundesliga"),
        ("soccer", "Serie A"),
        ("soccer", "Ligue 1"),
        ("soccer", "Premier League Women"),
        ("soccer", "La Liga U23"),
        ("basketball", "NBA"),
        ("basketball", "EuroLeague"),
        ("tennis", "ATP Tour"),
        ("hockey", "NHL"),
    ]
    
    teams = {
        "Premier League": [
            ("Arsenal", "Chelsea"), ("Manchester United", "Liverpool"),
            ("Tottenham Hotspur", "Manchester City"), ("Aston Villa", "Newcastle United"),
            ("Brighton and Hove Albion", "West Ham United"), ("Everton", "Fulham"),
        ],
        "La Liga": [
            ("Barcelona", "Real Madrid"), ("Atletico de Madrid", "Sevilla"),
            ("Real Sociedad", "Villarreal"), ("Athletic Club", "Celta Vigo"),
        ],
        "Bundesliga": [
            ("Bayern Munich", "Borussia Dortmund"), ("RB Leipzig", "Bayer Leverkusen"),
            ("Wolfsburg", "Eintracht Frankfurt"),
        ],
        "Serie A": [
            ("Juventus", "AC Milan"), ("Inter Milan", "Napoli"),
            ("Roma", "Lazio"),
        ],
        "Ligue 1": [
            ("Paris Saint-Germain", "Marseille"), ("Lyon", "Monaco"),
        ],
        "Premier League Women": [
            ("Arsenal Women", "Chelsea Women"), ("Man City Women", "Liverpool Women"),
        ],
        "La Liga U23": [
            ("Barcelona B", "Real Madrid Castilla"), ("Atletico B", "Sevilla Atletico"),
        ],
        "NBA": [
            ("LA Lakers", "Boston Celtics"), ("Golden State Warriors", "Miami Heat"),
            ("Milwaukee Bucks", "Denver Nuggets"),
        ],
        "EuroLeague": [
            ("Real Madrid", "Barcelona"), ("Fenerbahce", "Olympiacos"),
        ],
        "ATP Tour": [
            ("Djokovic N.", "Alcaraz C."), ("Sinner J.", "Medvedev D."),
        ],
        "NHL": [
            ("Toronto Maple Leafs", "Montreal Canadiens"),
            ("New York Rangers", "Boston Bruins"),
        ],
    }
    
    # B365 alias variations (simulating platform differences)
    b365_aliases = {
        "Manchester United": "Man Utd",
        "Manchester City": "Man City",
        "Tottenham Hotspur": "Spurs",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "Atletico de Madrid": "Atletico Madrid",
        "Paris Saint-Germain": "PSG",
        "Borussia Dortmund": "Dortmund",
        "Bayern Munich": "Bayern Munchen",
        "Inter Milan": "Internazionale",
        "Arsenal Women": "Arsenal (W)",
        "Chelsea Women": "Chelsea (W)",
        "Man City Women": "Manchester City (W)",
        "Liverpool Women": "Liverpool (W)",
        "Barcelona B": "FC Barcelona II",
        "Real Madrid Castilla": "Real Madrid II",
        "Atletico B": "Atletico Madrid B",
        "LA Lakers": "Los Angeles Lakers",
        "Golden State Warriors": "GS Warriors",
        "Toronto Maple Leafs": "Tor Maple Leafs",
    }
    
    op_matches = []
    b365_matches = []
    ground_truth = []  # (op_id, b365_id, decision)
    
    base_time = datetime(2025, 1, 15, 14, 0, 0)
    
    for i in range(n_matches):
        sport, league = random.choice(leagues)
        available_teams = teams.get(league, [("Team A", "Team B")])
        home, away = random.choice(available_teams)
        kickoff = base_time + timedelta(hours=i * 2, minutes=random.randint(-5, 5))
        
        # Detect categories
        full_text = f"{league} {home} {away}"
        cats = detect_categories(full_text)
        
        # OP match
        op = MatchRecord(
            match_id=f"OP_{i:04d}",
            platform="OP",
            sport=sport,
            league=league,
            home_team=home,
            away_team=away,
            kickoff=kickoff,
            category_tags=cats,
        )
        op_matches.append(op)
        
        # B365 match (with aliases and slight time drift)
        b365_home = b365_aliases.get(home, home)
        b365_away = b365_aliases.get(away, away)
        b365_kickoff = kickoff + timedelta(minutes=random.randint(-3, 3))
        
        b365 = MatchRecord(
            match_id=f"B365_{i:04d}",
            platform="B365",
            sport=sport,
            league=league,
            home_team=b365_home,
            away_team=b365_away,
            kickoff=b365_kickoff,
            category_tags=cats,
        )
        b365_matches.append(b365)
        
        # Ground truth: 85% match, 5% swapped, 10% no-match
        r = random.random()
        if r < 0.85:
            decision = Decision.MATCH
            gt_b365_id = b365.match_id
        elif r < 0.90:
            decision = Decision.SWAPPED
            gt_b365_id = b365.match_id
        else:
            decision = Decision.NO_MATCH
            gt_b365_id = None
        
        ground_truth.append((op.match_id, gt_b365_id, decision))
    
    # Add some extra B365 matches (distractors)
    for i in range(50):
        sport, league = random.choice(leagues)
        available_teams = teams.get(league, [("Team X", "Team Y")])
        home, away = random.choice(available_teams)
        
        b365_matches.append(MatchRecord(
            match_id=f"B365_EXTRA_{i:04d}",
            platform="B365",
            sport=sport,
            league=league,
            home_team=home + " FC",
            away_team=away + " United",
            kickoff=base_time + timedelta(hours=i * 3),
            category_tags=detect_categories(f"{league} {home} {away}"),
        ))
    
    return op_matches, b365_matches, ground_truth


# ═══════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  AI MATCH MAPPING ENGINE — END-TO-END DEMO")
    print("=" * 70 + "\n")
    
    # ── Step 1: Generate Data ──
    print("📊 Step 1: Generating synthetic match data...")
    op_matches, b365_matches, ground_truth = generate_synthetic_data(200)
    print(f"   OP matches:   {len(op_matches)}")
    print(f"   B365 matches: {len(b365_matches)}")
    print(f"   Ground truth: {len(ground_truth)}")
    
    gt_map = {op_id: (b365_id, dec) for op_id, b365_id, dec in ground_truth}
    
    # ── Step 2: Initialize Engine + Index B365 ──
    print("\n🔧 Step 2: Initializing inference engine...")
    engine = InferenceEngine()
    
    print("   Indexing B365 pool with SBERT...")
    engine.index_b365_pool(b365_matches)
    print(f"   ✓ Indexed {len(b365_matches)} B365 matches")
    
    # ── Step 3: Run Inference ──
    print("\n🔍 Step 3: Running inference on first 20 OP matches...")
    sample = op_matches[:20]
    
    for op in sample[:5]:
        suggestion = engine.predict(op)
        b365_id, dec = gt_map[op.match_id]
        
        print(f"\n   OP: {op.home_team} vs {op.away_team} ({op.league})")
        print(f"   Gate: {suggestion.gate_decision.value}")
        print(f"   Ground Truth: {dec.value}" + (f" → {b365_id}" if b365_id else ""))
        
        for c in suggestion.candidates_top5[:3]:
            marker = "✓" if c.b365_match_id == b365_id else " "
            print(f"   {marker} #{c.rank}: {c.b365_home} vs {c.b365_away} "
                  f"(score={c.score:.4f}, Δt={c.time_diff_minutes / 60.0:.1f}h, swap={c.swapped})")
    
    # ── Step 4: Simulate Feedback Ingestion ──
    print("\n💬 Step 4: Ingesting human feedback (simulated)...")
    store = FeedbackStore()
    
    for op in sample:
        suggestion = engine.predict(op)
        store.store_suggestion(suggestion)
        
        b365_id, decision = gt_map[op.match_id]
        
        fb = FeedbackRecord(
            mapping_suggestion_id=suggestion.mapping_suggestion_id,
            op_match_id=op.match_id,
            decision=decision,
            selected_b365_match_id=b365_id,
            swapped=(decision == Decision.SWAPPED),
        )
        success, msg = store.ingest_feedback(fb)
    
    counts = store.get_feedback_count()
    print(f"   Feedback: {counts}")
    print(f"   Training pairs: {len(store.get_training_pairs())}")
    print(f"     Positives:      {len(store.get_positives())}")
    print(f"     Hard negatives: {len(store.get_hard_negatives())}")
    
    # ── Step 5: Validate Training Data ──
    print("\n✅ Step 5: Validating training data...")
    pairs = store.get_training_pairs()
    is_clean = DatasetBuilder.validate_no_contamination(pairs)
    print(f"   Label contamination check: {'PASS ✓' if is_clean else 'FAIL ✗'}")
    
    # ── Step 6: Offline Evaluation (Baseline) ──
    print("\n📈 Step 6: Running offline evaluation (baseline)...")
    eval_data = []
    for op in op_matches[:100]:  # Evaluate on 100 samples
        b365_id, decision = gt_map[op.match_id]
        eval_data.append(EvalDataPoint(
            op_match=op,
            true_b365_id=b365_id,
            decision=decision,
            is_swapped=(decision == Decision.SWAPPED),
        ))
    
    evaluator = Evaluator(engine)
    result = evaluator.evaluate(eval_data, model_version="baseline-v1")
    
    # Generate report
    os.makedirs("reports", exist_ok=True)
    report = ReportGenerator.generate_report(
        result,
        output_path="reports/baseline_eval.txt",
    )
    print(report)
    
    # ── Step 7: Summary ──
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — Summary")
    print("=" * 70)
    print(f"""
  ✓ Inference engine: SBERT retrieval → Cross-Encoder reranking → Top-5
  ✓ Auto-match gates: MinScore + Margin + Category + Kickoff
  ✓ Feedback ingestion: {len(store.get_training_pairs())} training pairs generated
  ✓ Label contamination: {'Clean ✓' if is_clean else 'Contaminated ✗'}
  ✓ Baseline evaluation:
      Recall@5:              {result.recall_at_5:.4f}
      Recall@10:             {result.recall_at_10:.4f}
      Precision@1:           {result.precision_at_1:.4f}
      AUTO_MATCH Precision:  {result.auto_match_precision:.4f}
      AUTO_MATCH Rate:       {result.auto_match_rate:.4f}
      No-Match FP Rate:      {result.no_match_false_positive_rate:.4f}
    """)
    
    print("  Next steps:")
    print("  1. Load 10k real labels → BulkFeedbackLoader.load_from_records()")
    print("  2. Train models → TrainingOrchestrator.train_all()")
    print("  3. Deploy behind feature flag → POST /models/reload")
    print("  4. Compare before/after → ReportGenerator with previous_result")
    print("  5. Monitor drift weekly → per-sport/per-category metrics")
    print()


if __name__ == "__main__":
    main()

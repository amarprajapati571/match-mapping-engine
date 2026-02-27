"""
Alias-Based Training Pipeline — Fine-tune SBERT + Cross-Encoder from league/team data.

Generates synthetic training pairs from admin_leagues.json and admin_teams.json
so the ML models learn to recognize abbreviations, keywords, and name variations.

How it works:
  For each league/team with keywords or abbreviation:
    1. POSITIVE pair: match text with keyword ↔ match text with canonical name
       → Model learns: "CBA" and "Chinese Basketball Association" are the same
    2. HARD NEGATIVE: match text with one team ↔ match text with DIFFERENT team
       → Model learns: "Inter San Carlos" is NOT "Real Espana"

Training pair format (same as CSE feedback training):
  anchor_text:    "league | home vs away"  (using keyword/abbreviation)
  candidate_text: "league | home vs away"  (using canonical name)
  label: 1.0 (positive) or 0.0 (negative)

Usage:
    python scripts/train_from_aliases.py
    python scripts/train_from_aliases.py --leagues-file data/admin_leagues.json --teams-file data/admin_teams.json
    python scripts/train_from_aliases.py --dry-run
    python scripts/train_from_aliases.py --max-pairs 5000
    python scripts/train_from_aliases.py --sbert-only
    python scripts/train_from_aliases.py --ce-only
"""

import sys
import os
import json
import random
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.models import TrainingPair
from core.normalizer import build_match_text, detect_categories, clean_text
from core.league_loader import load_leagues_from_file, build_alias_map
from core.team_loader import load_teams_from_file, build_team_alias_map
from core.feedback import FeedbackStore
from training.trainer import TrainingOrchestrator, DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_from_aliases")


# ── Synthetic match templates for diverse training data ──
SPORTS = ["soccer", "basketball", "tennis", "cricket", "baseball", "hockey"]

DUMMY_TEAMS = [
    ("Team Alpha", "Team Beta"),
    ("Club United", "Club City"),
    ("FC Phoenix", "FC Dragon"),
    ("Warriors", "Knights"),
    ("Eagles", "Lions"),
    ("Thunder", "Storm"),
]


def generate_league_pairs(
    leagues: list,
    max_pairs_per_league: int = 5,
    seed: int = 42,
) -> list:
    """
    Generate positive training pairs from league abbreviations and keywords.

    For each league that has abbreviation or keywords different from canonical:
      - Creates pairs where one side uses abbreviation/keyword, the other uses canonical.
      - Uses random dummy teams so the model learns league matching independent of teams.

    Returns list of TrainingPair objects.
    """
    rng = random.Random(seed)
    pairs = []

    for league in leagues:
        league_name = league.get("league_name_en", "").strip()
        country = league.get("country_en", "").strip()
        abbreviation = league.get("league_abbreviation", "").strip()
        keywords = league.get("keywords", [])

        if not league_name:
            continue

        # Canonical league text
        canonical_league = f"{country} {league_name}" if country else league_name

        # Collect all variants (abbreviation + keywords)
        variants = set()
        if abbreviation and clean_text(abbreviation) != clean_text(canonical_league):
            variants.add(abbreviation)
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                if clean_text(kw) != clean_text(canonical_league):
                    variants.add(kw.strip())

        if not variants:
            continue

        # Generate pairs: variant league ↔ canonical league (same teams)
        variant_list = list(variants)
        for variant in variant_list[:max_pairs_per_league]:
            home, away = rng.choice(DUMMY_TEAMS)
            sport = rng.choice(SPORTS)

            cats_canon = detect_categories(f"{canonical_league} {home} {away}")
            cats_variant = detect_categories(f"{variant} {home} {away}")

            anchor = build_match_text(variant, home, away, cats_variant)
            candidate = build_match_text(canonical_league, home, away, cats_canon)

            pairs.append(TrainingPair(
                anchor_text=anchor,
                candidate_text=candidate,
                label=1.0,
                is_hard_negative=False,
                source_suggestion_id=f"league_alias_{clean_text(league_name)[:30]}",
            ))

    logger.info(f"Generated {len(pairs)} league alias training pairs")
    return pairs


def generate_team_pairs(
    teams: list,
    max_pairs_per_team: int = 3,
    seed: int = 42,
) -> list:
    """
    Generate positive training pairs from team keywords.

    For each team with keywords different from canonical name:
      - Creates pairs where one side uses the keyword, the other uses canonical.
      - Uses a fixed league context so the model learns team matching.

    Returns list of TrainingPair objects.
    """
    rng = random.Random(seed)
    pairs = []
    leagues_pool = [
        "Premier League", "La Liga", "Serie A", "Bundesliga",
        "Ligue 1", "Champions League", "MLS", "J-League",
    ]

    for team in teams:
        team_name = (
            team.get("team_name_en", "")
            or team.get("name_en", "")
        ).strip()
        keywords = team.get("keywords", [])

        if not team_name:
            continue

        # Collect variants that differ from canonical
        variants = set()
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                if clean_text(kw) != clean_text(team_name):
                    variants.add(kw.strip())

        if not variants:
            continue

        # Generate pairs: keyword team ↔ canonical team (same league, same opponent)
        variant_list = list(variants)
        for variant in variant_list[:max_pairs_per_team]:
            league = rng.choice(leagues_pool)
            opponent = rng.choice(DUMMY_TEAMS)[0]  # Random opponent

            cats = detect_categories(f"{league} {team_name} {opponent}")

            # Pair 1: variant as home team
            anchor = build_match_text(league, variant, opponent, cats)
            candidate = build_match_text(league, team_name, opponent, cats)

            pairs.append(TrainingPair(
                anchor_text=anchor,
                candidate_text=candidate,
                label=1.0,
                is_hard_negative=False,
                source_suggestion_id=f"team_alias_{clean_text(team_name)[:30]}",
            ))

    logger.info(f"Generated {len(pairs)} team alias training pairs")
    return pairs


def generate_hard_negatives(
    teams: list,
    n_negatives: int = 5000,
    seed: int = 42,
) -> list:
    """
    Generate hard negative pairs by pairing random DIFFERENT teams.

    This teaches the model that different teams should NOT match,
    even if they're in the same league.

    Returns list of TrainingPair objects.
    """
    rng = random.Random(seed)
    pairs = []

    # Collect team names
    team_names = []
    for team in teams:
        name = (
            team.get("team_name_en", "")
            or team.get("name_en", "")
        ).strip()
        if name:
            team_names.append(name)

    if len(team_names) < 10:
        logger.warning("Not enough teams for hard negative generation")
        return []

    leagues_pool = [
        "Premier League", "La Liga", "Serie A", "Bundesliga",
        "Ligue 1", "Champions League", "MLS", "J-League",
        "IPL", "NBA", "ATP Tour", "NHL",
    ]

    for _ in range(n_negatives):
        league = rng.choice(leagues_pool)

        # Pick 4 different teams
        selected = rng.sample(team_names, min(4, len(team_names)))
        home_a, away_a = selected[0], selected[1]
        home_b, away_b = selected[2] if len(selected) > 2 else selected[1], selected[3] if len(selected) > 3 else selected[0]

        cats = detect_categories(f"{league} {home_a} {away_a}")

        anchor = build_match_text(league, home_a, away_a, cats)
        candidate = build_match_text(league, home_b, away_b, cats)

        pairs.append(TrainingPair(
            anchor_text=anchor,
            candidate_text=candidate,
            label=0.0,
            is_hard_negative=True,
            source_suggestion_id=f"neg_{_}",
        ))

    logger.info(f"Generated {len(pairs)} hard negative training pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune models from league/team alias data"
    )
    parser.add_argument(
        "--leagues-file", default="data/admin_leagues.json",
        help="Path to leagues JSON file (default: data/admin_leagues.json)",
    )
    parser.add_argument(
        "--teams-file", default="data/admin_teams.json",
        help="Path to teams JSON file (default: data/admin_teams.json)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Limit total training pairs (random sample if exceeded)",
    )
    parser.add_argument(
        "--max-league-variants", type=int, default=5,
        help="Max keyword variants per league (default: 5)",
    )
    parser.add_argument(
        "--max-team-variants", type=int, default=3,
        help="Max keyword variants per team (default: 3)",
    )
    parser.add_argument(
        "--hard-negatives", type=int, default=0,
        help="Number of hard negative pairs to generate (default: auto-balance with positives)",
    )
    parser.add_argument(
        "--sbert-only", action="store_true",
        help="Only train SBERT bi-encoder",
    )
    parser.add_argument(
        "--ce-only", action="store_true",
        help="Only train Cross-Encoder reranker",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate pairs but don't train — shows stats only",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.epochs:
        CONFIG.training.sbert_epochs = args.epochs
        CONFIG.training.ce_epochs = args.epochs
    if args.batch_size:
        CONFIG.training.sbert_batch_size = args.batch_size
        CONFIG.training.ce_batch_size = args.batch_size

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("  ALIAS-BASED TRAINING — League/Team Data → Model Improvement")
    print("=" * 70 + "\n")

    # ── Step 1: Load Data ──
    print("Step 1: Loading league and team data...")

    leagues = load_leagues_from_file(args.leagues_file)
    teams = load_teams_from_file(args.teams_file)

    print(f"  Leagues loaded: {len(leagues)}")
    print(f"  Teams loaded:   {len(teams)}")

    if not leagues and not teams:
        print("\n  ERROR: No data found. Check file paths.")
        print(f"  Leagues file: {args.leagues_file}")
        print(f"  Teams file:   {args.teams_file}")
        return

    # ── Step 2: Generate Training Pairs ──
    print("\nStep 2: Generating training pairs from aliases...")

    league_pairs = generate_league_pairs(
        leagues,
        max_pairs_per_league=args.max_league_variants,
        seed=args.seed,
    )
    team_pairs = generate_team_pairs(
        teams,
        max_pairs_per_team=args.max_team_variants,
        seed=args.seed,
    )

    # Auto-balance: generate enough negatives to match positives (1:1 ratio)
    total_positives = len(league_pairs) + len(team_pairs)
    n_negatives = args.hard_negatives if args.hard_negatives > 0 else total_positives
    print(f"  Generating {n_negatives} hard negatives (to balance {total_positives} positives)...")

    neg_pairs = generate_hard_negatives(
        teams,
        n_negatives=n_negatives,
        seed=args.seed,
    )

    all_pairs = league_pairs + team_pairs + neg_pairs
    positives = [p for p in all_pairs if p.label == 1.0]
    negatives = [p for p in all_pairs if p.label == 0.0]

    # Optional: limit total pairs
    if args.max_pairs and len(all_pairs) > args.max_pairs:
        rng = random.Random(args.seed)
        # Keep ratio of positives to negatives
        pos_ratio = len(positives) / len(all_pairs)
        n_pos = int(args.max_pairs * pos_ratio)
        n_neg = args.max_pairs - n_pos
        sampled_pos = rng.sample(positives, min(n_pos, len(positives)))
        sampled_neg = rng.sample(negatives, min(n_neg, len(negatives)))
        all_pairs = sampled_pos + sampled_neg
        positives = sampled_pos
        negatives = sampled_neg
        print(f"  Sampled down to {len(all_pairs)} pairs (--max-pairs {args.max_pairs})")

    print(f"\n  Training pair summary:")
    print(f"    League alias pairs (positive):  {len(league_pairs)}")
    print(f"    Team alias pairs (positive):    {len(team_pairs)}")
    print(f"    Hard negatives:                 {len(neg_pairs)}")
    print(f"    ─────────────────────────────────")
    print(f"    Total pairs:                    {len(all_pairs)}")
    print(f"    Positives:                      {len(positives)}")
    print(f"    Negatives:                      {len(negatives)}")

    if positives and negatives:
        ratio = min(len(positives), len(negatives)) / max(len(positives), len(negatives))
        print(f"    Class ratio (min/max):          {ratio:.2f}")

    # Show some examples
    print(f"\n  Sample POSITIVE pairs:")
    for p in positives[:3]:
        print(f"    anchor:    {p.anchor_text[:80]}")
        print(f"    candidate: {p.candidate_text[:80]}")
        print(f"    label: {p.label}")
        print()

    print(f"  Sample NEGATIVE pairs:")
    for p in negatives[:2]:
        print(f"    anchor:    {p.anchor_text[:80]}")
        print(f"    candidate: {p.candidate_text[:80]}")
        print(f"    label: {p.label}")
        print()

    # ── Step 3: Validate ──
    print("Step 3: Validating training data...")
    if not DatasetBuilder.validate_no_contamination(all_pairs):
        print("  ABORTING: Label contamination detected!")
        sys.exit(1)
    print("  Data validation passed.")

    if args.dry_run:
        print("\n  --dry-run: Skipping training. Data looks good!")
        return

    # ── Step 4: Train Models ──
    print("\nStep 4: Training models from alias data...")
    os.makedirs("models", exist_ok=True)

    sbert_out = f"models/sbert_alias_tuned_{timestamp}"
    ce_out = f"models/ce_alias_tuned_{timestamp}"

    orchestrator = TrainingOrchestrator(all_pairs)
    stats = orchestrator.prepare()

    if args.ce_only:
        print("  Training Cross-Encoder only...")
        ce_path = orchestrator.train_cross_encoder(ce_out)
        if ce_path:
            print(f"  Cross-Encoder saved: {ce_path}")
        else:
            print("  Cross-Encoder training skipped — no examples.")
    elif args.sbert_only:
        print("  Training SBERT only...")
        sbert_path = orchestrator.train_sbert(sbert_out)
        if sbert_path:
            print(f"  SBERT saved: {sbert_path}")
        else:
            print("  SBERT training skipped — no data available.")
    else:
        print("  Training both SBERT + Cross-Encoder...")
        result = orchestrator.train_all(sbert_out, ce_out, track_accuracy=True)
        sbert_out = result.get("sbert_model_path") or "SKIPPED"
        ce_out = result.get("cross_encoder_model_path") or "SKIPPED"
        if result.get("accuracy_comparison"):
            print(result["accuracy_comparison"])

    # ── Step 5: Save Report ──
    os.makedirs("reports", exist_ok=True)
    report = {
        "timestamp": timestamp,
        "source": "alias_training",
        "leagues_file": args.leagues_file,
        "teams_file": args.teams_file,
        "leagues_count": len(leagues),
        "teams_count": len(teams),
        "league_pairs": len(league_pairs),
        "team_pairs": len(team_pairs),
        "hard_negatives": len(neg_pairs),
        "total_pairs": len(all_pairs),
        "positives": len(positives),
        "negatives": len(negatives),
        "sbert_path": sbert_out,
        "ce_path": ce_out,
    }
    report_path = os.path.join("reports", f"alias_train_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ALIAS-BASED TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
  Source Data:
    Leagues:              {len(leagues)} (from {args.leagues_file})
    Teams:                {len(teams)} (from {args.teams_file})

  Training Pairs:
    League alias pairs:   {len(league_pairs)}
    Team alias pairs:     {len(team_pairs)}
    Hard negatives:       {len(neg_pairs)}
    Total:                {len(all_pairs)}

  What the model learned:
    ✓ League abbreviations (CBA → Chinese Basketball Association)
    ✓ League keywords (J2 → Japan J2 League)
    ✓ Team name variations (FC Motagua → Motagua)
    ✓ Different teams should NOT match (hard negatives)

  Model artifacts:
    SBERT:         {sbert_out if not args.ce_only else 'N/A'}
    Cross-Encoder: {ce_out if not args.sbert_only else 'N/A'}
    Report:        {report_path}

  To deploy the new models:
    curl -X POST http://localhost:8000/models/reload \\
      -H "Content-Type: application/json" \\
      -d '{{"use_tuned_sbert": true, "use_tuned_cross_encoder": true, \\
           "tuned_sbert_path": "{sbert_out}", \\
           "tuned_cross_encoder_path": "{ce_out}"}}'
""")


if __name__ == "__main__":
    main()

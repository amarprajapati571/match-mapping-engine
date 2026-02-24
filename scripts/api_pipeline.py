"""
Live API Integration Pipeline.

Fetches real match data from the sports-bet API, runs AI mapping inference,
and outputs suggested mappings to a JSON file for review.

Performance:
  - Concurrent Bet365 + OddsPortal fetching via ThreadPoolExecutor
  - Batch-optimized inference (single SBERT encode + single CE scoring call)

Run: python scripts/api_pipeline.py
"""

import sys
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG
from core.inference import InferenceEngine

from scripts.pipeline_utils import (
    fetch_all_pages,
    submit_provider_fetch,
    convert_to_match_record,
    suggestion_to_dict,
    build_match_summary,
    push_results_to_api,
    save_json,
    CONFIDENCE_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("api_pipeline")

BET365_ENDPOINT = CONFIG.endpoints.bet365_endpoint
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ═══════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════

def _print_match_summary(summary: dict, provider_name: str = ""):
    """Print a human-readable match summary."""
    header = f"  MATCH SUMMARY — {provider_name}" if provider_name else "  MATCH SUMMARY"
    print("")
    print("─" * 50)
    print(header)
    print("─" * 50)
    print(f"  Total OP matches processed:  {summary['total_processed']}")
    print("")

    gd = summary["gate_decisions"]
    print(f"  Gate Decisions:")
    print(f"    AUTO_MATCH:   {gd['AUTO_MATCH']}")
    print(f"    NEED_REVIEW:  {gd['NEED_REVIEW']}")
    print("")

    bm = summary["bet365_match"]
    print(f"  B365 Match Status:")
    print(f"    Matched (has bet365_match):    {bm['matched']}")
    print(f"    No match (bet365_match null):  {bm['no_match_null']}")
    print("")

    cd = summary["confidence_distribution"]
    print(f"  Confidence Distribution:")
    print(f"    High   (>= 0.80):  {cd['high_gte_080']}")
    print(f"    Medium (0.50-0.80): {cd['medium_050_080']}")
    print(f"    Low    (< 0.50):    {cd['low_lt_050']}")
    print(f"    No candidate:       {cd['no_candidate']}")
    print("")

    if summary["by_sport"]:
        print(f"  By Sport:")
        for sport, sc in sorted(summary["by_sport"].items(), key=lambda x: x[1]["total"], reverse=True):
            print(
                f"    {sport:<20s}  total={sc['total']:<5d} "
                f"auto={sc['auto_match']:<5d} review={sc['need_review']:<5d} "
                f"no_cand={sc['no_candidate']}"
            )
    print("─" * 50)
    print("")


# ═══════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live API → AI Mapping Pipeline (Multi-Provider)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Reuse previously fetched data from data/*.json instead of calling APIs")
    args = parser.parse_args()

    providers = CONFIG.endpoints.get_active_providers()
    provider_names = [p["name"] for p in providers]

    print("\n" + "=" * 70)
    print("  LIVE API → AI MAPPING PIPELINE (MULTI-PROVIDER)")
    print("=" * 70)
    print(f"  Providers: {', '.join(provider_names)}")
    print("=" * 70 + "\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1-2: Fetch all providers ──
    # ODDSPORTAL: Bet365 from separate API  |  Others: Bet365 bundled in same API
    fetch_data: dict = {}
    needs_separate_bet365 = [p for p in providers if p.get("separate_bet365")]

    if args.skip_fetch:
        print("Step 1-2: Loading cached data (--skip-fetch)...")
        for p in providers:
            name = p["name"]
            path = os.path.join(DATA_DIR, f"{name.lower()}_raw.json")
            try:
                with open(path, "r") as f:
                    fetch_data[name] = json.load(f)
            except FileNotFoundError:
                fetch_data[name] = []
            b365_path_prov = os.path.join(DATA_DIR, f"{name.lower()}_bet365_raw.json")
            try:
                with open(b365_path_prov, "r") as f:
                    b365_data = json.load(f)
                    if b365_data:
                        fetch_data[f"{name}_bet365"] = b365_data
            except FileNotFoundError:
                pass
            b365_count = len(fetch_data.get(f'{name}_bet365', []))
            b365_info = f" (+{b365_count} B365)" if b365_count else ""
            print(f"  {name}: {len(fetch_data[name])} matches{b365_info}")
    else:
        print(f"Step 1-2: Fetching {len(providers)} providers concurrently...")
        if needs_separate_bet365:
            sep_names = [p["name"] for p in needs_separate_bet365]
            print(f"  Separate Bet365 fetch for: {', '.join(sep_names)}")

        with ThreadPoolExecutor(max_workers=len(providers) + 1) as executor:
            futures = {}

            if needs_separate_bet365:
                futures[executor.submit(
                    fetch_all_pages, BET365_ENDPOINT, "Bet365"
                )] = ("_shared_bet365", None)

            for p in providers:
                futures[submit_provider_fetch(executor, p)] = (p["name"], p)

            for future in as_completed(futures):
                label, provider_info = futures[future]
                try:
                    raw = future.result()

                    if label == "_shared_bet365":
                        shared_b365 = raw["matches"]
                        for p in needs_separate_bet365:
                            fetch_data[f"{p['name']}_bet365"] = shared_b365
                        print(f"  [Bet365] fetched {len(shared_b365)} matches (for {', '.join(p['name'] for p in needs_separate_bet365)})")
                        continue

                    fetch_data[label] = raw["matches"]
                    if raw.get("bet365"):
                        fetch_data[f"{label}_bet365"] = raw["bet365"]
                        print(
                            f"  [{label}] {len(raw['matches'])} provider matches "
                            f"+ {len(raw['bet365'])} bundled Bet365 matches"
                        )
                    else:
                        print(f"  [{label}] {len(raw['matches'])} provider matches")

                except Exception as e:
                    logger.error(f"Failed to fetch {label}: {e}")
                    if label != "_shared_bet365":
                        fetch_data[label] = []

        for label, rows in fetch_data.items():
            if isinstance(rows, list):
                save_json(rows, os.path.join(DATA_DIR, f"{label.lower()}_raw.json"))

    # ── Step 3-4: Init engine and run per-provider inference ──
    print(f"\nStep 3: Initializing AI inference engine...")
    engine = InferenceEngine()

    current_b365_source = None

    # ── Step 4: Run inference per provider (each uses its own Bet365 pool) ──
    all_pushable: list = []
    provider_stats: dict = {}
    total_processed = 0
    total_null_skipped = 0

    for p in providers:
        name = p["name"]
        raw = fetch_data.get(name, [])
        if not raw:
            print(f"\n  [{name}] No data — skipping.")
            provider_stats[name] = {"total": 0, "pushable": 0}
            continue

        provider_b365_raw = fetch_data.get(f"{name}_bet365", [])

        if not provider_b365_raw:
            print(f"\n  [{name}] No Bet365 data in API response — skipping.")
            provider_stats[name] = {"total": 0, "pushable": 0}
            continue

        b365_records = [convert_to_match_record(r, "B365") for r in provider_b365_raw]
        b365_source = f"{name}_bet365"
        print(f"\n  [{name}] Using Bet365 from {name} API: {len(b365_records)} matches")

        if b365_source != current_b365_source:
            print(f"  Indexing {len(b365_records)} B365 matches (source: {name} API)...")
            engine.index_b365_pool(b365_records)
            current_b365_source = b365_source

        records = [convert_to_match_record(r, name) for r in raw]
        print(f"\n  Step 4 [{name}]: Running inference on {len(records)} matches...")

        start = time.time()
        suggestions = engine.predict_batch(records)
        elapsed = time.time() - start
        rate = len(records) / max(elapsed, 0.001)
        print(f"  Inference complete in {elapsed:.1f}s ({rate:.1f} matches/s)")

        all_results = [suggestion_to_dict(s, platform=name) for s in suggestions]
        total_processed += len(all_results)

        summary = build_match_summary(suggestions, all_results)
        _print_match_summary(summary, provider_name=name)

        output = [r for r in all_results if r["confidence"] >= CONFIDENCE_THRESHOLD and r.get("bet365_match")]
        null_skipped = sum(1 for r in all_results if r["confidence"] >= CONFIDENCE_THRESHOLD and not r.get("bet365_match"))
        total_null_skipped += null_skipped
        all_pushable.extend(output)

        provider_stats[name] = {
            "total": len(all_results),
            "pushable": len(output),
            "null_skipped": null_skipped,
            "rate": round(rate, 1),
            "summary": summary,
        }

    # ── Step 6: Save and push combined results ──
    print(f"\nStep 6: Filtering combined results...")
    print(f"  Total processed (all providers): {total_processed}")
    print(f"  Total pushable:                  {len(all_pushable)}")
    print(f"  Total null bet365_match skipped: {total_null_skipped}")

    output_path = None
    if CONFIG.output.save_to_file and all_pushable:
        print(f"\nStep 6a: Saving {len(all_pushable)} results to file...")
        output_path = os.path.join(DATA_DIR, "ai_suggested_mappings.json")
        save_json(all_pushable, output_path)
    elif CONFIG.output.save_to_file:
        print("\nStep 6a: No pushable results to save.")
    else:
        print("\nStep 6a: SAVE_OUTPUT_TO_FILE=false — skipping local file save.")

    push_stats = None
    if CONFIG.output.push_to_api and all_pushable:
        print(f"\nStep 6b: Pushing {len(all_pushable)} results to API (parallel)...")
        push_stats = push_results_to_api(all_pushable)
    elif CONFIG.output.push_to_api:
        print("\nStep 6b: No pushable results (all null bet365_match or below threshold).")
    else:
        print("\nStep 6b: PUSH_RESULTS_TO_API=false — skipping API push.")

    # ── Step 7: Final report ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — MULTI-PROVIDER")
    print("=" * 70)
    print(f"\n  Data fetched:")
    for name in provider_names:
        b365_count = len(fetch_data.get(f"{name}_bet365", []))
        b365_info = f" (+{b365_count} Bet365)" if b365_count else " (no Bet365)"
        print(f"    {name}: {len(fetch_data.get(name, []))} matches{b365_info}")

    print(f"\n  Per-Provider Results:")
    for name, stats in provider_stats.items():
        print(f"    {name:<15s}  total={stats['total']:<6d} pushable={stats['pushable']:<6d} null_skipped={stats.get('null_skipped', 0)}")
    print(f"    {'TOTAL':<15s}  total={total_processed:<6d} pushable={len(all_pushable)}")

    print(f"\n  Output:")
    print(f"    Save to file: {'ON' if CONFIG.output.save_to_file else 'OFF'}{f' -> {output_path}' if output_path else ''}")
    print(f"    Push to API:  {'ON' if CONFIG.output.push_to_api else 'OFF'}{f' -> {push_stats}' if push_stats else ''}")

    print(f"\n  Sample output (first 5):")
    print("  " + "-" * 66)
    for r in all_pushable[:5]:
        print(f"  {json.dumps(r)}")
    print()


if __name__ == "__main__":
    main()

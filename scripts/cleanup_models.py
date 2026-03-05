"""
Cleanup useless single-class model files.

Scans the models/ directory for CSE-tuned models trained on single-class data
(only negatives, 0 positives) and removes them. These models are useless for
production because they always predict "no match".

Uses the accuracy_history.jsonl log to identify which runs had both classes.

Usage:
    python scripts/cleanup_models.py                  # Dry run (list what would be deleted)
    python scripts/cleanup_models.py --delete          # Actually delete
    python scripts/cleanup_models.py --delete --keep-latest 3  # Keep latest 3 good models
"""

import sys
import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CONFIG

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cleanup")

# Models to ALWAYS keep (known good combined runs)
ALWAYS_KEEP = {
    "sbert_cse_tuned_20260223_061258",
    "ce_cse_tuned_20260223_061258",
    "sbert_cse_tuned_20260227_095502",
    "ce_cse_tuned_20260227_095502",
    "sbert_cse_tuned_20260305_153707",
    "ce_cse_tuned_20260305_153707",
    # Alias models
    "sbert_alias_tuned_20260227_114306",
    "ce_alias_tuned_20260227_114306",
}


def find_good_runs_from_logs() -> set:
    """
    Parse accuracy_history.jsonl to find training runs that had BOTH classes.
    Returns set of model directory names that should be kept.
    """
    log_path = os.path.join(CONFIG.logs_dir, "accuracy_history.jsonl")
    if not os.path.exists(log_path):
        logger.warning(f"No accuracy log found at {log_path}")
        return set()

    good_model_dirs = set()

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                dataset = entry.get("dataset_info", {})
                total_pos = dataset.get("total_positives", 0)
                total_neg = dataset.get("total_negatives", 0)

                # Only keep models from runs with BOTH classes
                if total_pos > 0 and total_neg > 0:
                    paths = entry.get("model_paths", {})
                    for path in paths.values():
                        if path and isinstance(path, str):
                            dirname = os.path.basename(path)
                            good_model_dirs.add(dirname)
            except json.JSONDecodeError:
                continue

    return good_model_dirs


def scan_model_dirs(models_dir: str) -> dict:
    """
    Scan the models directory and categorize each model.
    Returns dict of {dir_name: {"path": ..., "type": ..., "keep": bool, "reason": str}}
    """
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        return {}

    good_runs = find_good_runs_from_logs()
    results = {}

    for entry in sorted(os.listdir(models_dir)):
        full_path = os.path.join(models_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # Only process cse_tuned and alias_tuned models
        is_cse = "cse_tuned" in entry
        is_alias = "alias_tuned" in entry
        if not (is_cse or is_alias):
            continue

        info = {
            "path": full_path,
            "type": "alias" if is_alias else "cse",
            "keep": False,
            "reason": "",
        }

        # Check always-keep list
        if entry in ALWAYS_KEEP:
            info["keep"] = True
            info["reason"] = "always-keep (known good model)"
        # Check if it was from a good run (both classes)
        elif entry in good_runs:
            info["keep"] = True
            info["reason"] = "good run (both classes in accuracy log)"
        # Alias models are always kept
        elif is_alias:
            info["keep"] = True
            info["reason"] = "alias model (separate pipeline)"
        else:
            info["reason"] = "single-class or unknown run"

        results[entry] = info

    return results


def get_dir_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def main():
    parser = argparse.ArgumentParser(
        description="Clean up useless single-class model files"
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete files (default is dry run)",
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Path to models directory (default: models/)",
    )
    parser.add_argument(
        "--keep-latest", type=int, default=0,
        help="Keep the N most recent CSE models regardless of quality",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL CLEANUP — Remove Useless Single-Class Models")
    print("=" * 70)
    if not args.delete:
        print("  MODE: DRY RUN (use --delete to actually remove files)")
    else:
        print("  MODE: DELETE")
    print()

    results = scan_model_dirs(args.models_dir)

    if not results:
        print("  No model directories found.")
        return

    # If --keep-latest, mark the N most recent CSE models as keep
    if args.keep_latest > 0:
        cse_models = [
            (name, info) for name, info in results.items()
            if info["type"] == "cse"
        ]
        # Sort by name (which includes timestamp) descending
        cse_models.sort(key=lambda x: x[0], reverse=True)
        for i, (name, info) in enumerate(cse_models[:args.keep_latest]):
            if not info["keep"]:
                results[name]["keep"] = True
                results[name]["reason"] = f"kept (--keep-latest {args.keep_latest})"

    # Categorize
    to_keep = {k: v for k, v in results.items() if v["keep"]}
    to_delete = {k: v for k, v in results.items() if not v["keep"]}

    print(f"  Total model directories: {len(results)}")
    print(f"  Keep:   {len(to_keep)}")
    print(f"  Delete: {len(to_delete)}")
    print()

    if to_keep:
        print("  KEEPING:")
        for name, info in sorted(to_keep.items()):
            print(f"    {name:<50} ({info['reason']})")
        print()

    if to_delete:
        total_size = 0
        print("  DELETING:" if args.delete else "  WOULD DELETE:")
        for name, info in sorted(to_delete.items()):
            size = get_dir_size(info["path"])
            total_size += size
            print(f"    {name:<50} {format_size(size):>8}  ({info['reason']})")
        print()
        print(f"  Total space to reclaim: {format_size(total_size)}")
        print()

        if args.delete:
            for name, info in sorted(to_delete.items()):
                try:
                    shutil.rmtree(info["path"])
                    print(f"    Deleted: {name}")
                except Exception as e:
                    print(f"    FAILED to delete {name}: {e}")
            print()
            print(f"  Cleanup complete. Removed {len(to_delete)} model directories.")
        else:
            print("  To delete these files, run again with --delete")
    else:
        print("  Nothing to clean up — all models are from good runs.")

    print()


if __name__ == "__main__":
    main()

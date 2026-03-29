#!/bin/bash
# P3: Cleanup bug-period model files (Feb 23 — Mar 5, 2026)
# These ~140 files were trained with zero positives and are useless.
#
# SAFE: Preserves the Feb 27 alias baseline model.
# Run with --dry-run first to see what would be deleted.

set -euo pipefail

MODELS_DIR="${1:-models}"
DRY_RUN="${2:-}"

if [ ! -d "$MODELS_DIR" ]; then
    echo "Models directory not found: $MODELS_DIR"
    exit 1
fi

echo "=== Bug-Period Model Cleanup ==="
echo "Directory: $MODELS_DIR"
echo ""

# Bug period: Feb 23 (20260223) through Mar 5 (20260305)
# EXCLUDE: alias model from Feb 27 (sbert_alias_tuned_20260227, ce_alias_tuned_20260227)
BUG_PATTERNS=(
    "*_cse_tuned_202602*"  # All CSE-tuned Feb models (bug period)
    "*_tuned_20260220*"    # Pre-bug test models
    "*_cse_tuned_20260301*"
    "*_cse_tuned_20260302*"
    "*_cse_tuned_20260303*"
    "*_cse_tuned_20260304*"
    "*_cse_tuned_20260305*"
)

# Protected models (never delete)
PROTECTED_PATTERNS=(
    "*_alias_tuned_20260227*"  # Alias baseline — keep forever
)

count=0
size=0

for pattern in "${BUG_PATTERNS[@]}"; do
    for model_dir in "$MODELS_DIR"/$pattern; do
        [ -e "$model_dir" ] || continue

        # Check if protected
        is_protected=false
        for prot in "${PROTECTED_PATTERNS[@]}"; do
            if [[ "$(basename "$model_dir")" == $prot ]]; then
                is_protected=true
                break
            fi
        done

        if $is_protected; then
            echo "  SKIP (protected): $(basename "$model_dir")"
            continue
        fi

        dir_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
        count=$((count + 1))

        if [ "$DRY_RUN" = "--dry-run" ]; then
            echo "  WOULD DELETE: $(basename "$model_dir") ($dir_size)"
        else
            echo "  DELETING: $(basename "$model_dir") ($dir_size)"
            rm -rf "$model_dir"
        fi
    done
done

echo ""
echo "Total: $count model directories"
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Run without --dry-run to actually delete."
else
    echo "Cleanup complete."
fi

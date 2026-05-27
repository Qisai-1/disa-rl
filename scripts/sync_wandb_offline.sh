#!/bin/bash
# Sync all locally-cached wandb runs (from WANDB_MODE=offline) to wandb.ai.
# Run after training jobs finish if you want the runs visible in the web UI.
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

# Sync everything in the local wandb cache
echo "Searching for offline runs to sync..."
for wandb_root in ./wandb /work/mech-ai-scratch/supersai/.wandb/wandb; do
    [ -d "$wandb_root" ] || continue
    echo "=== $wandb_root ==="
    n=$(ls -d "$wandb_root"/offline-run-* 2>/dev/null | wc -l)
    echo "  $n offline runs found"
    if [ "$n" -gt 0 ]; then
        wandb sync --include-offline "$wandb_root"/offline-run-* || \
        wandb sync "$wandb_root"/offline-run-*
    fi
done
echo "Done."

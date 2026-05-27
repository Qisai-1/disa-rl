#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Evaluate every saved IQL checkpoint and dump a results CSV.
#
#  Prereqs (run once on the GPU node):
#      pip install gymnasium "mujoco<3" imageio
#
#  Usage:
#      bash scripts/eval_all_checkpoints.sh           # eval all envs/alphas
#      bash scripts/eval_all_checkpoints.sh hopper-medium-v2 0.5
# ──────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/supersai/.conda/envs/disa

# Sanity: gymnasium must be importable
python -c "import gymnasium, mujoco" 2>/dev/null || {
    echo "ERROR: gymnasium / mujoco not installed in this env."
    echo "Fix:  pip install gymnasium 'mujoco<3' imageio"
    exit 1
}

ENVS=${1:-"halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 \
          halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2 ant-medium-replay-v2"}
# 0.5  = v3 baseline (offline_only) + v3 DRC (augmented)
# 0.15 = Tier-1 ablation (augmented only) — eval_checkpoints.py harmlessly
#        skips offline_only/alpha0.15 since that dir does not exist.
ALPHAS=${2:-"0.5 0.15"}
MODES=${3:-"offline_only augmented"}

mkdir -p results
OUT="results/eval_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "env,mode,alpha,seed,best_score,best_step,final_score" > "$OUT"

for env in $ENVS; do
    for mode in $MODES; do
        for alpha in $ALPHAS; do
            echo "=== $env  mode=$mode  alpha=$alpha ==="
            python eval_checkpoints.py --env "$env" --alpha "$alpha" --mode "$mode" \
                --n_episodes 10 2>&1 | tee -a "results/eval_$(date +%Y%m%d).log"
        done
    done
done

echo ""
echo "Summary written to: $OUT"

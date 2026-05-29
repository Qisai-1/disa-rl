#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stable-config offline-RL comparison — the FIRST protocol-compliant run.
#
#  Fixes baked in (from the 2026-05-28 debugging):
#    - reward_scale = 1  (reward_scale 10 caused Q-loss divergence + collapse)
#    - vanilla IQL base: no PA-loss, no action-noise (those were PARS add-ons)
#    - expectile 0.7, temperature 3.0 (published IQL locomotion settings)
#    - data = Fix-B (velocity-integration + analytic-done) physics-consistent syn
#
#  Protocol: {offline_only, Tier-1, CAPA} × {hopper, walker2d} × 3 seeds,
#  500k steps, eval every 10k (10 episodes). Report = mean of last-10 evals
#  ± std over seeds (scripts/aggregate_results.py → last10avg).
#  --save_every huge: we only need the eval logs, not checkpoints.
#
#  NOTE: eval is still gymnasium v4 (env mismatch vs d4rl v2) — so absolute
#  numbers aren't comparable to published tables, but the INTERNAL comparison
#  (offline vs Tier-1 vs CAPA, all in v4) is fair. d4rl-v2 eval is a separate fix.
#
#  USAGE: bash scripts/launch_stable_comparison.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: a python iql/train_iql.py is already running."; exit 1
fi
for e in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    [ -f "data/synthetic/$e/synthetic_transitions.npz" ] || { echo "FATAL: missing syn data for $e"; exit 1; }
done

mkdir -p logs
COMMON="--num_steps 500000 --save_every 999999999 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 --q_hidden_dims 256 256 \
        --wandb_project disa-rl-stable"
AUG="--mode augmented --alpha 0.15 --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"

i=0
run () {  # env method seed
  local env=$1 method=$2 seed=$3 gpu=$(( i % 4 )); i=$((i+1))
  case $method in
    offline) flags="--mode offline_only --num_critics 2 --critic_subset 2" ;;
    tier1)   flags="$AUG --num_critics 2 --critic_subset 2" ;;
    capa)    flags="$AUG --capa --unc_beta 1.0 --num_critics 10 --critic_subset 2" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $flags $COMMON > "logs/cmp_${env%%-*}_${method}_s${seed}.log" 2>&1 &
  echo "  GPU$gpu  $env  $method  seed=$seed"
}

echo "=== stable comparison launching $(date '+%F %T') ==="
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for method in offline tier1 capa; do
    for seed in 0 1 2; do
      run "$env" "$method" "$seed"
    done
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-stable')" -ge 18 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-stable')/18 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "aggregate with: python scripts/aggregate_results.py --logs_glob 'logs/cmp_*.log'"

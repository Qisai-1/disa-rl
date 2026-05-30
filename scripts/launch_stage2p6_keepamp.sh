#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2.6: KEEP the GTA amplification.
#
#  Diagnosis (2026-05-29, after Stage 2.5 launched): SyntheticBuffer was
#  per-row renormalizing syn rewards to match real's mean and std exactly
#  (iql/buffer.py:283-286, controlled by normalize_rewards=True). For ordinary
#  syn data this protects against syn-reward bias. For GTA-amplified data it
#  WIPES OUT the very signal we generated: syn 100-step return mean was 262
#  (hopper) / 293 (walker), real was 237 / 247 — a +10/+19% lift erased by
#  the renorm.
#
#  Fix: added --no-syn_normalize_rewards flag (default still True for backward
#  compat). Stage 2.5 ran with the old behavior; Stage 2.6 runs with the new
#  flag. If 2.6 > 2.5, the amplification was the missing piece.
#
#  Config: same as Stage 2.5 (CORL + obs norm) + --no-syn_normalize_rewards.
#
#  USAGE:
#    NGPU=3 bash scripts/launch_stage2p6_keepamp.sh <ALPHA> <COEF>
#    e.g.  NGPU=3 bash scripts/launch_stage2p6_keepamp.sh 0.5 0.25
#
#  Output:
#    logs/s2p6_<env>_<method>_s<SEED>.log
#    wandb project: disa-rl-s2p6
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

ALPHA=${1:-0.5}
COEF=${2:-0.25}
NGPU=${NGPU:-4}

if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: iql/train_iql.py running (GPUs busy). Wait for current sweep."; exit 1
fi
for e in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    [ -f "data/synthetic_gta/$e/synthetic_transitions.npz" ] || {
        echo "FATAL: missing GTA syn data for $e"; exit 1
    }
done

mkdir -p logs
COMMON="--num_steps 500000 --save_every 100000 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 \
        --reward_norm corl --obs_norm --no-syn_normalize_rewards \
        --q_hidden_dims 256 256 --wandb_project disa-rl-s2p6"
AUG="--mode augmented --alpha ${ALPHA} --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef ${COEF} \
      --num_critics 10 --critic_subset 2"

i=0
run () {  # env method seed
  local env=$1 method=$2 seed=$3 gpu=$(( i % NGPU )); i=$((i+1))
  local tag="${env%%-*}_${method}_s${seed}"
  case $method in
    capa+)   flags="$AUG $CAPA --synthetic_data data/synthetic_gta/${env}/synthetic_transitions.npz" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $flags $COMMON > "logs/s2p6_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  $method  seed=$seed"
}

echo "=== Stage 2.6 (keep-amplification) launching $(date '+%F %T') ==="
echo "    alpha=${ALPHA}  critic_syn_coef=${COEF}"
# Only CAPA+ — the offline baseline doesn't touch syn data, no point re-running it.
# Stage 2.5's offline_only numbers carry over.
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for method in capa+; do
    for seed in 0 1 2; do
      run "$env" "$method" "$seed"
    done
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s2p6')" -ge 6 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s2p6')/6 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

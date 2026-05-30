#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2.5: CORL reward-normalization ablation.
#
#  Diagnosis (2026-05-29): our depressed offline baseline (hopper 29.4 vs
#  published ~95) is partly the v4 eval discount but ALSO that we use raw
#  D4RL rewards (per-step mean ~2.4 on hopper) instead of the standard CORL
#  /IQL recipe (rewards × 1000/(max_ep_ret-min_ep_ret) → per-step mean ~0.74).
#  Our temperature β=3 and expectile τ=0.7 were tuned for the lower scale →
#  the AWR weight exp(β·adv) becomes too peaky → policy underfits.
#
#  Test: --reward_norm corl AND --obs_norm (the full standard CORL/IQL recipe)
#  on stable config × {offline_only, CAPA+GTA-winner} × {hopper, walker} × 3
#  seeds. Combined fix expected to lift offline baseline substantially AND
#  improve CAPA+GTA (same reason: critic+policy losses both better-scaled).
#
#  USAGE:
#    bash scripts/launch_stage2p5_corlnorm.sh <ALPHA> <COEF>
#    e.g.  bash scripts/launch_stage2p5_corlnorm.sh 0.5 0.25
#
#  Output:
#    logs/s2p5_<env>_<method>_s<SEED>.log
#    wandb project: disa-rl-s2p5
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

ALPHA=${1:-0.5}
COEF=${2:-0.25}

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
        --reward_norm corl --obs_norm \
        --q_hidden_dims 256 256 --wandb_project disa-rl-s2p5"
AUG="--mode augmented --alpha ${ALPHA} --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef ${COEF} \
      --num_critics 10 --critic_subset 2"

i=0
run () {  # env method seed
  local env=$1 method=$2 seed=$3 gpu=$(( i % 4 )); i=$((i+1))
  local tag="${env%%-*}_${method}_s${seed}"
  case $method in
    offline) flags="--mode offline_only --num_critics 2 --critic_subset 2" ;;
    capa+)   flags="$AUG $CAPA --synthetic_data data/synthetic_gta/${env}/synthetic_transitions.npz" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $flags $COMMON > "logs/s2p5_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  $method  seed=$seed"
}

echo "=== Stage 2.5 (CORL norm) launching $(date '+%F %T') ==="
echo "    alpha=${ALPHA}  critic_syn_coef=${COEF}"
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for method in offline capa+; do
    for seed in 0 1 2; do
      run "$env" "$method" "$seed"
    done
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s2p5')" -ge 12 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s2p5')/12 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

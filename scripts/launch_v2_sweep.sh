#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  DiSA-RL v2 sweep — full DRC-IQL stack + PARS (reward scaling + PA loss)
#  + alpha warmup + REDQ full-min ensemble.
#
#  Run from the salloc shell on the GPU node, AFTER v1 sweep finishes.
#  Currently uses seed=0 only; expand to 5 seeds once v2 shows lift.
#
#  Settings tuned per-env from PARS configs:
#    halfcheetah, ant      : reward_scale=5
#    hopper, walker2d      : reward_scale=10
# ─────────────────────────────────────────────────────────────────────────────

cd /work/mech-ai-scratch/supersai/disa-rl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
export WANDB_MODE=offline
mkdir -p logs

reward_scale_for() {
  case "$1" in
    halfcheetah*|ant*) echo 5  ;;
    hopper*|walker2d*) echo 10 ;;
    *)                 echo 1  ;;
  esac
}

# critic_subset for REDQ-style min: 10 (full ensemble) on hopper/walker2d
# (which are termination-heavy and benefit from strong pessimism)
critic_subset_for() {
  case "$1" in
    hopper*|walker2d*) echo 10 ;;
    *)                 echo 2  ;;
  esac
}

ENVS_MEDIUM=( halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 )
ENVS_REPLAY=( halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2 ant-medium-replay-v2 )

launch_one() {
  local env=$1 mode=$2 gpu=$3 alpha=$4 tag=$5
  local rs=$(reward_scale_for "$env")
  local cs=$(critic_subset_for "$env")
  local extra=""
  if [[ "$mode" == "augmented" ]]; then
    extra="--mode augmented --alpha $alpha --bc_weight 0.1 \
           --sa_iql --expectile_real 0.9 --expectile_syn 0.5 --sa_clip 0.5 2.0 \
           --alpha_warmup 50000 --alpha_ramp 50000"
  else
    extra="--mode offline_only --bc_weight 0.0"
  fi

  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py \
      --env "$env" $extra --seed 0 \
      --num_steps 500000 \
      --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset $cs \
      --action_noise_std 0.05 \
      --reward_scale $rs --pa_weight 0.001 --pa_min_q -100 \
      --wandb_project "disa-rl-v2-${tag}" \
      > "logs/v2_${tag}_${env}_s0.log" 2>&1 &
}

# 4 envs × 3 method variants = 12 jobs, 3 per GPU
for i in 0 1 2 3; do
  ENV=${ENVS_MEDIUM[$i]}
  launch_one "$ENV" augmented    $i 0.5 "drc"      # full DRC-IQL + PARS bits
  launch_one "$ENV" offline_only $i ""  "baseline" # IQL + PARS bits, no syn
done
# Also: medium-replay envs with full DRC + PARS
for i in 0 1 2 3; do
  ENV=${ENVS_REPLAY[$i]}
  launch_one "$ENV" augmented $i 0.5 "drc-replay"
done

sleep 5
echo "=== launched $(pgrep -f train_iql.py | wc -l) v2 processes at $(date) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

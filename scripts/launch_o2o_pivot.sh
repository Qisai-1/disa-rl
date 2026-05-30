#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  O2O pivot launcher — the namesake offline-to-online phase.
#
#  Composition:
#    - SAC initialized from offline IQL/CAPA checkpoint (load_from_iql)
#    - Conservative blend decays 1→0 over transition_steps
#    - Critic-only warmup for critic_warmup_steps
#    - Online real interactions go into RealEnvBuffer
#    - Mixed sampling: ρ blends offline syn (GTA-amplified) + fresh syn + real
#    - Diffusion model fine-tunes online on fresh real data (EWC-regularized)
#    - εroll (MMD-based) adaptively shrinks ρ as real distribution shifts
#
#  Acts as the FINAL fallback if Stages 2-4 don't yield a clean offline win.
#  Per the namesake paper, the O2O gain is the headline result —
#  not the offline numbers — when real env interaction breaks the
#  data-processing limit.
#
#  USAGE:
#    bash scripts/launch_o2o_pivot.sh
#
#  ENV-specific IQL ckpt path is hardcoded below — update if a fresher
#  CAPA+ ckpt is available (Stages 2-4 use --save_every 999999999, so they
#  need to be re-run with --save_every enabled to provide an O2O init).
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

# Refuse if GPUs are busy.
if pgrep -f 'python -u (iql|online_rl)/' >/dev/null; then
    echo "ERROR: offline RL or online RL is already running (GPUs busy)."
    exit 1
fi

# Best available IQL ckpts (stale; update when fresh CAPA+ ckpts available).
declare -A IQL_CKPT=(
    [hopper-medium-replay-v2]="checkpoints/hopper-medium-replay-v2/iql/augmented/alpha0.15/seed_2/best.pt"
    [walker2d-medium-replay-v2]="checkpoints/walker2d-medium-replay-v2/iql/capa/alpha0.5/seed_0/best.pt"
)

mkdir -p logs

i=0
run () {  # env seed
  local env=$1 seed=$2 gpu=$(( i % 4 )); i=$((i+1))
  local iql_ckpt=${IQL_CKPT[$env]}
  local dif_ckpt="checkpoints/${env}/diffusion/offline_final.pt"
  local syn="data/synthetic_gta/${env}/synthetic_transitions.npz"
  [ -f "$iql_ckpt" ] || { echo "MISSING IQL ckpt: $iql_ckpt"; exit 1; }
  [ -f "$dif_ckpt" ] || { echo "MISSING diffusion ckpt: $dif_ckpt"; exit 1; }
  [ -f "$syn" ]      || { echo "MISSING GTA syn data: $syn"; exit 1; }

  CUDA_VISIBLE_DEVICES=$gpu python -u online_rl/train_online.py \
    --env "$env" --seed "$seed" \
    --iql_ckpt "$iql_ckpt" \
    --diffusion_ckpt "$dif_ckpt" \
    --synthetic_data "$syn" \
    --num_steps 200000 \
    --critic_warmup_steps 5000 \
    --transition_steps 20000 \
    --eval_every 5000 --log_every 1000 \
    > "logs/o2o_${env%%-*}_s${seed}.log" 2>&1 &
  echo "  GPU$gpu  $env  seed=$seed  iql=${iql_ckpt}"
}

echo "=== O2O pivot launching $(date '+%F %T') ==="
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for seed in 0 1 2; do
    run "$env" "$seed"
  done
done

until [ "$(pgrep -fc 'online_rl/train_online.py')" -ge 6 ]; do sleep 3; done
echo "launched $(pgrep -fc 'online_rl/train_online.py')/6 O2O runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

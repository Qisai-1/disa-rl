#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4: regenerate GTA syn data with stronger return amplification
#  (--gta_return_alpha 1.5 vs the default 1.2), then run the winning
#  CAPA+/UTD config on the new data.
#
#  Rationale (per [[reference-literature-aaai]]): amplified-return guidance
#  drives generative stitching toward higher-return compositions. α=1.2 gave
#  our first clean win; α=1.5 stretches the conditioning further — verify it
#  doesn't push the diffusion off-manifold (velocity-integration + analytic
#  termination should hold).
#
#  USAGE:
#    bash scripts/launch_stage4_alpha15.sh <ALPHA_SAMPLE> <COEF> [UTD]
#    e.g.  bash scripts/launch_stage4_alpha15.sh 0.5 0.25 4
#
#  Steps:
#    1. Generate hopper+walker syn data with --gta_return_alpha 1.5 into
#       data/synthetic_gta_a15/ (if not already present).
#    2. Audit residuals (sanity print obs std + reward stats).
#    3. Launch CAPA+ training × 3 seeds × {hopper, walker} on the new data.
#
#  Output:
#    data/synthetic_gta_a15/<env>/synthetic_transitions.npz
#    logs/s4_<env>_a<ALPHA_SAMPLE>_c<COEF>_u<UTD>_s<SEED>.log
#    wandb project: disa-rl-s4
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

ALPHA_SAMPLE=${1:-0.5}
COEF=${2:-0.25}
UTD=${3:-4}

# Refuse if GPUs are already busy with stage 2/3.
if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: a python iql/train_iql.py is already running (GPUs busy)."
    exit 1
fi

OUT=data/synthetic_gta_a15
mkdir -p "$OUT" logs

# ── 1. Regen syn data with α=1.5 ─────────────────────────────────────────────
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  if [ ! -f "$OUT/$env/synthetic_transitions.npz" ]; then
    echo "=== regenerating $env GTA syn data (α=1.5) $(date '+%F %T') ==="
    python -u generate_synthetic_data.py --env "$env" \
        --n_transitions 1000000 \
        --integrate_velocity \
        --gta --gta_noise_ratio 0.5 --gta_return_alpha 1.5 \
        --output_dir "$OUT" \
        > "logs/s4_gen_${env%%-*}.log" 2>&1
    echo "  wrote $OUT/$env/synthetic_transitions.npz"
  else
    echo "  $env GTA-α1.5 syn data already exists, skipping regen."
  fi
done

# Quick audit
python - <<PY
import numpy as np, os
for e in ("hopper-medium-replay-v2","walker2d-medium-replay-v2"):
    p=f"$OUT/{e}/synthetic_transitions.npz"
    d=np.load(p, allow_pickle=True)
    print(f"  {e}: N={len(d['rewards']):>8}  r_mean={d['rewards'].mean():.3f}  r_std={d['rewards'].std():.3f}  obs_std_mean={d['observations'].std(0).mean():.3f}")
PY

# ── 2. Launch CAPA+ training ────────────────────────────────────────────────
COMMON="--num_steps 500000 --save_every 999999999 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 --q_hidden_dims 256 256 \
        --wandb_project disa-rl-s4"
AUG="--mode augmented --alpha ${ALPHA_SAMPLE} --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef ${COEF} \
      --num_critics 10 --critic_subset 2 --utd ${UTD}"

i=0
run () {  # env seed
  local env=$1 seed=$2 gpu=$(( i % 4 )); i=$((i+1))
  local tag="${env%%-*}_a${ALPHA_SAMPLE}_c${COEF}_u${UTD}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $AUG $CAPA \
      --synthetic_data "$OUT/${env}/synthetic_transitions.npz" \
      $COMMON > "logs/s4_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  seed=$seed  alpha=$ALPHA_SAMPLE coef=$COEF utd=$UTD"
}

echo "=== Stage 4 (GTA α=1.5) launching $(date '+%F %T') ==="
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for seed in 0 1 2; do
    run "$env" "$seed"
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s4')" -ge 6 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s4')/6 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

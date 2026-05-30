#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2.7 ABLATIONS — leave-one-out from the best combo.
#
#  Only run AFTER Stage 2.7 lands a decisive win. Each row of the ablation
#  table drops one novelty knob to isolate its contribution.
#
#  Conditions (each × {hopper, walker} × 3 seeds = 24 runs total):
#    A) ALL novelties              (Stage 2.7 — already run, results carry over)
#    B) drop awr_gate_mode=temper  → back to scale
#    C) drop gbc_weight            → 0
#    D) drop asym_expectile_syn    → off
#    E) drop critic_syn_coef_warmup→ 0 (instant target)
#
#  Total: 4 × 2 × 3 = 24 runs (B/C/D/E only — A reused from 2.7).
#
#  GPU layout: 24 runs / 3 H200 = 8/GPU. That's heavy — split into two
#  waves of 12 (4/GPU each).
#
#  USAGE:
#    NGPU=3 bash scripts/launch_stage2p7_ablations.sh wave1   # B + C
#    (wait until done)
#    NGPU=3 bash scripts/launch_stage2p7_ablations.sh wave2   # D + E
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

WAVE=${1:-wave1}
NGPU=${NGPU:-4}

if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: iql/train_iql.py running. Wait."; exit 1
fi

mkdir -p logs
COMMON="--num_steps 500000 --save_every 100000 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 \
        --reward_norm corl --obs_norm --no-syn_normalize_rewards \
        --utd 4 \
        --q_hidden_dims 256 256 --wandb_project disa-rl-s2p7-abl"
AUG="--mode augmented --alpha 0.5 --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef 0.25 \
      --num_critics 10 --critic_subset 2"

# Per-ablation differentials from the full combo (Stage 2.7):
ABL_FULL="--awr_gate_mode temper --gbc_weight 0.05 --gbc_gate_min 0.6 \
          --asym_expectile_syn --critic_syn_coef_warmup 50000"
ABL_NO_TEMPER="                                  --gbc_weight 0.05 --gbc_gate_min 0.6 \
               --asym_expectile_syn --critic_syn_coef_warmup 50000"
ABL_NO_GBC="--awr_gate_mode temper                                       \
            --asym_expectile_syn --critic_syn_coef_warmup 50000"
ABL_NO_ASYM="--awr_gate_mode temper --gbc_weight 0.05 --gbc_gate_min 0.6 \
                                   --critic_syn_coef_warmup 50000"
ABL_NO_WARMUP="--awr_gate_mode temper --gbc_weight 0.05 --gbc_gate_min 0.6 \
               --asym_expectile_syn                                    "

case $WAVE in
  wave1) ABLS=("no_temper:$ABL_NO_TEMPER" "no_gbc:$ABL_NO_GBC") ;;
  wave2) ABLS=("no_asym:$ABL_NO_ASYM" "no_warmup:$ABL_NO_WARMUP") ;;
  *) echo "Unknown wave $WAVE (use wave1 or wave2)"; exit 1 ;;
esac

i=0
run() {  # env abl_tag flags seed
  local env=$1 tag=$2 seed=$3 gpu=$(( i % NGPU )); i=$((i+1))
  shift 3
  local flags="$@"
  local logtag="${env%%-*}_${tag}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $AUG $CAPA $flags $COMMON \
      --synthetic_data "data/synthetic_gta/${env}/synthetic_transitions.npz" \
      > "logs/s2p7abl_${logtag}.log" 2>&1 &
  echo "  GPU$gpu  $env  $tag  seed=$seed"
}

echo "=== Stage 2.7 ablations ($WAVE) launching $(date '+%F %T') ==="
for abl in "${ABLS[@]}"; do
  tag="${abl%%:*}"
  flags="${abl#*:}"
  for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    for seed in 0 1 2; do
      run "$env" "$tag" "$seed" $flags
    done
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s2p7-abl')" -ge 12 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s2p7-abl') / 12 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

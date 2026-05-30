#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2.7: BEST COMBO — full calibrated-stitching novelty pack.
#
#  Stacks every novelty addition on top of the Stage 2.5/2.6 baseline:
#    - CORL reward norm + obs norm                       (Stage 2.5 fix)
#    - --no-syn_normalize_rewards                        (Stage 2.6 fix; preserves GTA amplification)
#    - --awr_gate_mode temper                            (β-modulated AWR weights)
#    - --gbc_weight 0.05 --gbc_gate_min 0.6              (Generative BC anchor on high-conf syn)
#    - --asym_expectile_syn                              (τ_syn = 0.5 + 0.2·gate)
#    - --critic_syn_coef_warmup 50000                    (curriculum 0 → target over 50k)
#    - --utd 4                                           (RLPD-style UTD>1)
#
#  Diffusion-side novelties (adaptive α, Q-guidance, ensemble) need
#  regenerated data + Stage 2.5 checkpoints; deferred to Stage 3.x.
#
#  Comparison plan:
#    - Stage 2.5 offline_only       : baseline-with-fix    (carries over)
#    - Stage 2.5 CAPA+GTA           : CAPA+ with only the norm fixes
#    - Stage 2.6 CAPA+GTA           : + amplification preserved
#    - Stage 2.7 (this)             : + all RL-side novelties stacked
#  If 2.7 > 2.6 > 2.5 > baseline, the ablation table writes itself.
#
#  USAGE:
#    NGPU=3 bash scripts/launch_stage2p7_bestcombo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

NGPU=${NGPU:-4}
ALPHA=${ALPHA:-0.5}
COEF=${COEF:-0.25}
GBC=${GBC:-0.05}
WARMUP=${WARMUP:-50000}

if [ "${SKIP_GUARD:-0}" != "1" ] && pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: iql/train_iql.py running (GPUs busy). Pass SKIP_GUARD=1 to override."; exit 1
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
        --utd 4 \
        --q_hidden_dims 256 256 --wandb_project disa-rl-s2p7"
AUG="--mode augmented --alpha ${ALPHA} --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef ${COEF} \
      --num_critics 10 --critic_subset 2"
NOVEL="--awr_gate_mode temper --gbc_weight ${GBC} --gbc_gate_min 0.6 \
       --asym_expectile_syn --critic_syn_coef_warmup ${WARMUP}"

i=0
run () {  # env seed
  local env=$1 seed=$2 gpu=$(( i % NGPU )); i=$((i+1))
  local tag="${env%%-*}_bestcombo_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $AUG $CAPA $NOVEL $COMMON \
      --synthetic_data "data/synthetic_gta/${env}/synthetic_transitions.npz" \
      > "logs/s2p7_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  seed=$seed"
}

echo "=== Stage 2.7 (best-combo) launching $(date '+%F %T') ==="
echo "    alpha=${ALPHA}  coef=${COEF}  gbc=${GBC}  coef_warmup=${WARMUP}  utd=4"
# 6 runs (2 envs × 3 seeds) — offline_only from Stage 2.5 carries over.
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for seed in 0 1 2; do
    run "$env" "$seed"
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s2p7')" -ge 6 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s2p7')/6 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

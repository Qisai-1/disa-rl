#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Fill the A40s — pack 3 additional runs per GPU on top of the 4 CAPA-s0
#  runs already running (total 4 procs/GPU). User authorized aggressive GPU
#  utilization 2026-05-27 ~14:30 ("u dont need to save anything").
#
#  Per env (1 GPU each), adds:
#    - CAPA seed 1     →  variance estimate for headline method
#    - CAPA seed 2     →  3-seed CAPA mean (paper-grade)
#    - Tier-1 seed 0   →  fair-comparison Tier-1 baseline on NEW syn data
#                         (old Tier-1 ckpts archived to seed_0_OLDSYN_*)
#
#  --save_every 99999999  → no intermediate step_*.pt (best.pt + final.pt only)
#
#  USAGE:
#    bash scripts/launch_capa_fill_gpus.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

ENVS=( halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2 ant-medium-replay-v2 )

reward_scale_for() {
    case "$1" in
        halfcheetah*|ant*) echo 5  ;;
        hopper*|walker2d*) echo 10 ;;
    esac
}
critic_subset_for() {
    case "$1" in
        hopper*|walker2d*) echo 10 ;;
        *)                 echo 2  ;;
    esac
}

# Common flags identical to scripts/launch_capa_replay.sh / launch_tier1_ablation.sh
COMMON=( --mode augmented --alpha 0.15
         --num_steps 500000 --save_every 99999999 --bc_weight 0.1
         --q_hidden_dims 256 256 256 256 --num_critics 10
         --action_noise_std 0.05 --temperature 3.0
         --alpha_warmup 50000 --alpha_ramp 50000
         --pa_weight 0.001 --pa_min_q -100 )

unset WANDB_MODE
mkdir -p logs

echo "=== filling GPUs at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")

    # CAPA seed 1
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --capa --unc_beta 1.0 --seed 1 \
        --critic_subset "$cs" --reward_scale "$rs" \
        --wandb_project disa-rl-capa "${COMMON[@]}" \
        > "logs/capa_${ENV}_s1.log" 2>&1 &

    # CAPA seed 2
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --capa --unc_beta 1.0 --seed 2 \
        --critic_subset "$cs" --reward_scale "$rs" \
        --wandb_project disa-rl-capa "${COMMON[@]}" \
        > "logs/capa_${ENV}_s2.log" 2>&1 &

    # Tier-1 seed 0 (fair-comparison rerun on NEW syn data; old s0 archived)
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --seed 0 \
        --critic_subset "$cs" --reward_scale "$rs" \
        --wandb_project disa-rl-tier1-newsyn "${COMMON[@]}" \
        > "logs/t1new_${ENV}_s0.log" 2>&1 &

    echo "  GPU $i +3 procs ($ENV: CAPA s1, CAPA s2, Tier1 s0-newsyn)"
done

until [ "$(pgrep -fc 'iql/train_iql.py' || echo 0)" -ge 16 ]; do sleep 2; done
echo
echo "=== total iql/train_iql.py procs: $(pgrep -fc 'iql/train_iql.py') (target 16) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain #5: post-diffusion-ensemble → ensemble-filtered regen +
#  Stage 3.0 (CAPA+novelty pack on ensemble-filtered syn data).
#
#  This is the actual test of the diffusion-side novelty: does ensemble
#  disagreement filtering produce better syn data than raw GTA?
#
#  USAGE:
#    NGPU=3 nohup bash scripts/auto_chain_post_diffens.sh > logs/auto_chain_5.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain-5 armed; waiting for diffusion ensemble to launch + finish..."

while [ "$(pgrep -fc 'diffusion/train.py')" -eq 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] Diffusion ensemble detected; now waiting for completion..."

while [ "$(pgrep -fc 'diffusion/train.py')" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] Diffusion ensemble done."

# Verify all 6 expected ensemble checkpoints exist
ALL_OK=1
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    for seed in 0 1 2; do
        ckpt="checkpoints/$env/diffusion_ens_s${seed}/offline_final.pt"
        if [ ! -f "$ckpt" ]; then
            echo "MISSING: $ckpt"; ALL_OK=0
        fi
    done
done
if [ $ALL_OK -ne 1 ]; then
    echo "[$(date '+%F %T')] Some ensemble ckpts missing — abort"; exit 1
fi
echo "[$(date '+%F %T')] All 6 ensemble checkpoints present"

# Sanity: no procs alive
if pgrep -f 'python -u (iql|diffusion)/train.py' >/dev/null; then
    echo "[$(date '+%F %T')] Training procs still alive — abort"; exit 1
fi

# Step 1: regenerate syn data with ensemble disagreement filter
mkdir -p data/synthetic_ens_filtered
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    echo "[$(date '+%F %T')] Regenerating $env with ensemble-disagreement filter..."
    python scripts/regen_with_ensemble_filter.py \
        --env "$env" --K 3 --keep_quantile 0.5 --n_transitions 1000000 \
        > "logs/regen_ens_${env%%-*}.log" 2>&1
    echo "[$(date '+%F %T')] $env regen done"
done

# Step 2: launch Stage 3.0 — CAPA + novelty pack on ensemble-filtered data
# Same config as Stage 2.7 but pointing at synthetic_ens_filtered/
mkdir -p logs
COMMON="--num_steps 500000 --save_every 100000 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 \
        --reward_norm corl --obs_norm --no-syn_normalize_rewards \
        --utd 4 \
        --q_hidden_dims 256 256 --wandb_project disa-rl-s3p0"
AUG="--mode augmented --alpha 0.5 --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef 0.25 \
      --num_critics 10 --critic_subset 2"
NOVEL="--awr_gate_mode temper --gbc_weight 0.05 --gbc_gate_min 0.6 \
       --asym_expectile_syn --critic_syn_coef_warmup 50000"

i=0
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    for seed in 0 1 2; do
        gpu=$(( i % NGPU )); i=$((i+1))
        tag="${env%%-*}_ensfilt_s${seed}"
        CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
            $AUG $CAPA $NOVEL $COMMON \
            --synthetic_data "data/synthetic_ens_filtered/${env}/synthetic_transitions.npz" \
            > "logs/s3p0_${tag}.log" 2>&1 &
        echo "  GPU$gpu  $env  seed=$seed"
    done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s3p0')" -ge 6 ]; do sleep 3; done
echo "[$(date '+%F %T')] Stage 3.0 launched (6 runs); ETA ~5h"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Queue auto-chain-6 to evaluate Stage 3.0 + decide what's next
NGPU=$NGPU nohup bash scripts/auto_chain_post_s3p0.sh > logs/auto_chain_6.log 2>&1 &
echo "[$(date '+%F %T')] Auto-chain-6 (post-Stage-3.0) armed (PID $!)"

echo "[$(date '+%F %T')] === Auto-chain-5 complete ==="

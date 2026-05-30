#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Master chain — replaces the buggy multi-daemon setup with one script.
#
#  Uses a robust counter that excludes pgrep self-matches:
#      count_procs <wandb_project_substr>
#
#  Waits for current Stage 2.6/2.7 → TD3+BC → verdict gate → ablations +
#  diffusion ensemble → ensemble regen + Stage 3.0 → multi-seed extension.
#
#  USAGE:
#    NGPU=3 nohup bash scripts/master_chain.sh > logs/master_chain.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

count_procs() {  # $1 = wandb project substring
    ps -eo pid,cmd 2>/dev/null \
        | grep -E "python -u iql/train_iql\.py.*disa-rl-$1" \
        | grep -v grep | wc -l
}
count_diffusion_procs() {
    ps -eo pid,cmd 2>/dev/null \
        | grep -E "python -u diffusion/train\.py" \
        | grep -v grep | wc -l
}

wait_for_exit() {  # $1 = project substr, $2 = label
    echo "[$(date '+%F %T')] Waiting for $2 procs (project=$1) to launch+finish..."
    # Wait for launch
    while [ "$(count_procs $1)" -eq 0 ]; do sleep 30; done
    echo "[$(date '+%F %T')] $2 procs detected ($(count_procs $1) alive); waiting for completion..."
    while [ "$(count_procs $1)" -gt 0 ]; do sleep 60; done
    echo "[$(date '+%F %T')] $2 procs all exited."
}

aggregate() {  # $1 = log glob, $2 = output file
    python scripts/report_runs.py "$1" > "$2" 2>&1 || true
    echo "  --- $2 ---"
    tail -10 "$2"
}

mkdir -p logs results

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: wait for current Stage 2.6 + 2.7 to finish
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Master chain armed ==="
echo "[$(date '+%F %T')] Waiting on Stage 2.6 (project=s2p6) and Stage 2.7 (project=s2p7)..."

# Wait for BOTH to launch and BOTH to finish
while [ "$(count_procs s2p6)" -eq 0 ] && [ "$(count_procs s2p7)" -eq 0 ]; do
    sleep 30
done
echo "[$(date '+%F %T')] 2.6/2.7 procs detected"
while [ "$(count_procs s2p6)" -gt 0 ] || [ "$(count_procs s2p7)" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] 2.6 + 2.7 both done."

aggregate 'logs/s2p6_*.log' results/stage2p6_final.txt
aggregate 'logs/s2p7_*.log' results/stage2p7_final.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: TD3+BC sweep (18 runs, ~6h)
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Launching TD3+BC sweep ==="
SKIP_GUARD=1 NGPU=$NGPU bash scripts/launch_td3bc_sweep.sh || {
    echo "[$(date '+%F %T')] TD3+BC launcher failed — abort chain"; exit 1
}

wait_for_exit td3bc "TD3+BC"
aggregate 'logs/td3bc_*.log' results/td3bc_final.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 3: verdict check on Stage 2.7
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Checking Stage 2.7 verdict ==="
python scripts/verdict_check.py > results/stage2p7_verdict.txt 2>&1
cat results/stage2p7_verdict.txt | tail -25
VERDICT=$(grep -oE 'VERDICT=[A-Z]+' results/stage2p7_verdict.txt | tail -1 | cut -d= -f2)
echo "[$(date '+%F %T')] VERDICT=$VERDICT"

if [ "$VERDICT" = "LOSS" ]; then
    cat > results/post_td3bc_loss.txt <<EOF
Stage 2.7 verdict: LOSS — novelty pack did not beat the lifted Stage 2.5 baseline.
Vanilla CAPA+GTA regression on hopper (-42) was the deep hole; novelty pack
designed to recover but didn't. Possible next steps:
  - Lower alpha (0.15 instead of 0.5) — less syn mix
  - Drop GBC (adds syn influence, may hurt on strong baseline)
  - Drop CAPA+ critic-syn term — keep CAPA real-only
  - Try TD3+BC backbone first — might be more robust to syn noise
EOF
    cat results/post_td3bc_loss.txt
    echo "[$(date '+%F %T')] === Master chain stops on LOSS. ==="
    exit 0
fi

# ──────────────────────────────────────────────────────────────────────────
# Stage 4: leave-one-out ablations (WIN/MIXED only)
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Launching ablations wave1 (drop_temper + drop_gbc) ==="
SKIP_GUARD=1 NGPU=$NGPU bash scripts/launch_stage2p7_ablations.sh wave1 || {
    echo "[$(date '+%F %T')] Ablations wave1 launcher failed — skip ablations"
}

wait_for_exit s2p7-abl "Ablations wave1"
aggregate 'logs/s2p7abl_*.log' results/stage2p7_abl_wave1.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 5: diffusion ensemble training (6 runs, ~6h)
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Launching diffusion ensemble ==="
NGPU=$NGPU bash scripts/launch_diffusion_ensemble.sh || {
    echo "[$(date '+%F %T')] Diffusion ensemble launcher failed — abort"; exit 1
}

echo "[$(date '+%F %T')] Waiting on diffusion ensemble..."
while [ "$(count_diffusion_procs)" -eq 0 ]; do sleep 30; done
while [ "$(count_diffusion_procs)" -gt 0 ]; do sleep 60; done
echo "[$(date '+%F %T')] Diffusion ensemble done."

# Verify checkpoints
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    for seed in 0 1 2; do
        ckpt="checkpoints/$env/diffusion_ens_s${seed}/offline_final.pt"
        if [ ! -f "$ckpt" ]; then
            echo "[$(date '+%F %T')] Missing $ckpt — abort chain"; exit 1
        fi
    done
done

# ──────────────────────────────────────────────────────────────────────────
# Stage 6: ensemble-filter regen + Stage 3.0 (CAPA+novelty on filtered syn)
# ──────────────────────────────────────────────────────────────────────────
mkdir -p data/synthetic_ens_filtered
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    echo "[$(date '+%F %T')] Regen $env with ensemble disagreement filter..."
    python scripts/regen_with_ensemble_filter.py \
        --env "$env" --K 3 --keep_quantile 0.5 --n_transitions 1000000 \
        > "logs/regen_ens_${env%%-*}.log" 2>&1
done

# Launch Stage 3.0 — same combo as 2.7 but on filtered data
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
        CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
            $AUG $CAPA $NOVEL $COMMON \
            --synthetic_data "data/synthetic_ens_filtered/${env}/synthetic_transitions.npz" \
            > "logs/s3p0_${env%%-*}_ensfilt_s${seed}.log" 2>&1 &
    done
done
echo "[$(date '+%F %T')] Stage 3.0 launched (6 runs)"

wait_for_exit s3p0 "Stage 3.0"
aggregate 'logs/s3p0_*.log' results/stage3p0_final.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 7: multi-seed extension (5 seeds total on the best config)
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Multi-seed extension (seeds 3,4 to make 5 total) ==="
i=0
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    for seed in 3 4; do
        gpu=$(( i % NGPU )); i=$((i+1))
        CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
            $AUG $CAPA $NOVEL \
            --num_steps 500000 --save_every 100000 --eval_every 10000 \
            --expectile 0.7 --temperature 3.0 --reward_scale 1 \
            --reward_norm corl --obs_norm --no-syn_normalize_rewards \
            --utd 4 --q_hidden_dims 256 256 --wandb_project disa-rl-multiseed \
            --synthetic_data "data/synthetic_ens_filtered/${env}/synthetic_transitions.npz" \
            > "logs/ms_${env%%-*}_s${seed}.log" 2>&1 &
    done
done

wait_for_exit multiseed "Multi-seed"
aggregate 'logs/ms_*.log' results/multiseed_final.txt
aggregate 'logs/s3p0_*.log' results/stage3p0_final.txt

echo "[$(date '+%F %T')] === MASTER CHAIN COMPLETE ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

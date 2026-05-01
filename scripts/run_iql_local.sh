#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  DiSA-RL — Local IQL Training Script
#  Works on any single-GPU machine (titan3, scslab-titan1, beast0)
#
#  Usage:
#    bash scripts/run_iql_local.sh
#
#  Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these values
# ══════════════════════════════════════════════════════════════════════════════

# Environments to train (space-separated)
# titan3:        "halfcheetah-medium-v2 hopper-medium-v2"
# scslab-titan1: "walker2d-medium-v2 ant-medium-v2"
ENVS="halfcheetah-medium-v2 hopper-medium-v2"

# Alpha values to sweep (space-separated)
# 0.5  = 50% real + 50% synthetic
# 0.25 = 25% real + 75% synthetic
# 0.0  = 100% synthetic (pure diffusion)
ALPHAS="0.5 0.25 0.0"

# Seeds (space-separated)
SEEDS="0 1 2 3 4"

# Training mode
MODE="augmented"

# BC anchor weight (0.0 = disabled, 0.1 = recommended)
BC_WEIGHT=0.1

# WandB project
WANDB_PROJECT="disa-rl-medium"

# Max parallel seeds per env per alpha (tune for your GPU VRAM)
# 4090 24GB: 10 is safe  (5 per env × 2 envs)
# TitanX 12GB: 4 is safe (2 per env × 2 envs)
MAX_PARALLEL=10

# Total training steps
NUM_STEPS=1000000

# GPU index (usually 0, beast0 has 0 and 1)
GPU=0

# ══════════════════════════════════════════════════════════════════════════════
#  SCRIPT — no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."   # always run from repo root
mkdir -p logs

echo "=========================================="
echo "  DiSA-RL Local IQL Training"
echo "  $(date)"
echo "=========================================="
echo "  Envs    : $ENVS"
echo "  Alphas  : $ALPHAS"
echo "  Seeds   : $SEEDS"
echo "  Mode    : $MODE"
echo "  BC wt   : $BC_WEIGHT"
echo "  Project : $WANDB_PROJECT"
echo "  GPU     : $GPU"
echo "=========================================="
echo ""

# Verify synthetic data exists for augmented mode
if [[ "$MODE" == "augmented" ]]; then
    echo "Checking synthetic data..."
    for env in $ENVS; do
        syn="./data/synthetic/$env/synthetic_transitions.npz"
        if [[ -f "$syn" ]]; then
            echo "  ✓ $env"
        else
            echo "  ✗ $env — MISSING: $syn"
            echo "    Run: python generate_synthetic_data.py --env $env"
            exit 1
        fi
    done
    echo ""
fi

# Run alpha sweep — one alpha at a time, all envs×seeds in parallel
for alpha in $ALPHAS; do
    echo "------------------------------------------"
    echo "  Alpha = $alpha  ($(date))"
    echo "------------------------------------------"

    n_running=0
    for env in $ENVS; do
        for seed in $SEEDS; do
            # Wait if we've hit the parallel limit
            while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]]; do
                sleep 5
            done

            log="logs/iql_${env}_${MODE}_alpha${alpha}_s${seed}.log"
            echo "  Launching: $env alpha=$alpha seed=$seed → $log"

            CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
                --env            "$env" \
                --mode           "$MODE" \
                --alpha          "$alpha" \
                --bc_weight      "$BC_WEIGHT" \
                --seed           "$seed" \
                --num_steps      "$NUM_STEPS" \
                --wandb_project  "$WANDB_PROJECT" \
                >> "$log" 2>&1 &
        done
    done

    # Wait for all jobs at this alpha to finish
    wait
    echo "  Done alpha=$alpha  ($(date))"
    echo ""
done

echo "=========================================="
echo "  All done: $(date)"
echo "=========================================="

# Quick summary of final scores
echo ""
echo "Final scores:"
for env in $ENVS; do
    for alpha in $ALPHAS; do
        for seed in $SEEDS; do
            log="logs/iql_${env}_${MODE}_alpha${alpha}_s${seed}.log"
            score=$(grep "normalized=" "$log" 2>/dev/null | tail -1 | grep -oP "normalized=\K[0-9.]+")
            [[ -n "$score" ]] && echo "  $env alpha=$alpha seed=$seed → $score"
        done
    done
done

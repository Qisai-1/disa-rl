#!/usr/bin/env bash
# Run ablation studies on halfcheetah (representative environment)
# Ablations: fixed_alpha_0.3, fixed_alpha_0.5, fixed_alpha_0.7, offline_only

cd "$(dirname "$0")/.."

ENV="halfcheetah-medium-v2"
CKPT="./checkpoints/$ENV/diffusion/offline_final.pt"
SEEDS=(0 1 2)

# Fixed alpha ablations
for alpha in 0.3 0.5 0.7; do
    for seed in "${SEEDS[@]}"; do
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$ENV" --mode ablation_fixed_alpha \
            --diffusion_ckpt "$CKPT" \
            --alpha "$alpha" --seed "$seed"
    done
done

# No augmentation ablation
for seed in "${SEEDS[@]}"; do
    WANDB_MODE=offline python iql/train_iql.py \
        --env "$ENV" --mode offline_only --seed "$seed"
done

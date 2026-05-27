#!/usr/bin/env bash
# Train IQL baselines (no augmentation) across all envs and seeds
# Run this while diffusion models are training

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
SEEDS=(0 1 2 3 4)

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "IQL baseline: $env  seed=$seed"
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$env" \
            --mode offline_only \
            --seed "$seed" \
            --num_steps 1000000
    done
done

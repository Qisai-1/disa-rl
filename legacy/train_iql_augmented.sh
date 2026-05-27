#!/usr/bin/env bash
# Train DiSA-RL augmented IQL across all envs and seeds
# Run AFTER diffusion models are trained

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
SEEDS=(0 1 2 3 4)

for env in "${ENVS[@]}"; do
    CKPT="./checkpoints/$env/diffusion/offline_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping $env"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        echo "DiSA-RL augmented: $env  seed=$seed"
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$env" \
            --mode augmented \
            --diffusion_ckpt "$CKPT" \
            --seed "$seed" \
            --num_steps 1000000
    done
done

#!/usr/bin/env bash
# Train diffusion models for all environments
# Usage: bash scripts/train_diffusion.sh
# Runs two environments in parallel (adjust for your GPU count)

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
PIDS=()

for env in "${ENVS[@]}"; do
    echo "Starting diffusion training: $env"
    WANDB_MODE=offline python diffusion/train.py --env "$env" --batch_size 128 &
    PIDS+=($!)
    sleep 120   # Wait 2 min between launches to let VRAM stabilize
done

echo "All training jobs started. PIDs: ${PIDS[@]}"
wait
echo "All done."

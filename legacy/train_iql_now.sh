#!/bin/bash
# Simple IQL training script — just works
# Run from repo root: bash train_iql_now.sh

conda activate disa
cd ~/disa_rl

mkdir -p logs

GPU=0
WANDB_PROJECT="disa-rl-medium"

echo "Starting IQL training..."

# halfcheetah alpha=0.5
for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
        --env halfcheetah-medium-v2 \
        --mode augmented \
        --alpha 0.5 \
        --bc_weight 0.1 \
        --seed $seed \
        --num_steps 1000000 \
        --wandb_project $WANDB_PROJECT \
        > logs/halfcheetah_aug0.5_s${seed}.log 2>&1 &
done

# hopper alpha=0.5
for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
        --env hopper-medium-v2 \
        --mode augmented \
        --alpha 0.5 \
        --bc_weight 0.1 \
        --seed $seed \
        --num_steps 1000000 \
        --wandb_project $WANDB_PROJECT \
        > logs/hopper_aug0.5_s${seed}.log 2>&1 &
done

echo "Launched 10 jobs. Waiting..."
wait
echo "Done alpha=0.5"

# Check scores
grep "normalized=" logs/halfcheetah_aug0.5_s0.log | tail -3
grep "normalized=" logs/hopper_aug0.5_s0.log | tail -3

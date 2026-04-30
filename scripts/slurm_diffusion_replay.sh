#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name="disa_diffusion_replay"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/diffusion_replay_%j.out"
#SBATCH --error="logs/diffusion_replay_%j.err"

echo "=========================================="
echo "  DiSA-RL — Diffusion Training (replay)"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURMD_NODENAME"
echo "  Start:     $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

DISA=/work/mech-ai-scratch/supersai/disa-rl
cd $DISA
mkdir -p logs/slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/supersai/.conda/envs/disa

ENVS=(
    "halfcheetah-medium-replay-v2"
    "hopper-medium-replay-v2"
    "walker2d-medium-replay-v2"
    "ant-medium-replay-v2"
)

# Only 1 GPU — train sequentially
for ENV in "${ENVS[@]}"; do
    echo ""
    echo "Training: $ENV  ($(date))"
    python diffusion/train.py \
        --env        "$ENV" \
        --batch_size 256 \
        --lr         1e-4 \
        --patience   20 \
        --num_steps  300000 \
        >> "logs/slurm/diffusion_${ENV}_${SLURM_JOB_ID}.log" 2>&1

    echo "Generating synthetic data: $ENV"
    python generate_synthetic_data.py \
        --env       "$ENV" \
        --force_env "$ENV" \
        >> "logs/slurm/synthetic_${ENV}_${SLURM_JOB_ID}.log" 2>&1

    echo "Done: $ENV  ($(date))"
done

echo ""
echo "=========================================="
echo "  All replay environments complete: $(date)"
echo "=========================================="
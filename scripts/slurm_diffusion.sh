#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem=44G
#SBATCH --job-name="disa_diffusion"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/diffusion_%j.out"
#SBATCH --error="logs/diffusion_%j.err"

echo "=========================================="
echo "  DiSA-RL — Diffusion Model Training"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURMD_NODENAME"
echo "  Start:     $(date)"
echo "=========================================="

echo ""
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

DISA=/work/mech-ai-scratch/supersai/disa-rl
cd $DISA
mkdir -p logs/slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/supersai/.conda/envs/disa

# Train all 4 environments simultaneously, one per GPU
ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    echo "Launching diffusion training: $ENV on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python diffusion/train.py \
        --env        "$ENV" \
        --batch_size 256 \
        --lr         1e-4 \
        --patience   100 \
        --num_steps  300000 \
        >> "logs/slurm/diffusion_${ENV}_${SLURM_JOB_ID}.log" 2>&1 &
done

echo "All 4 diffusion jobs launched. Waiting..."
wait
echo "All diffusion training complete: $(date)"

# Generate synthetic data for all envs after training
echo ""
echo "Generating synthetic data..."
for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    echo "Generating: $ENV on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python generate_synthetic_data.py \
        --env       "$ENV" \
        --force_env "$ENV" \
        >> "logs/slurm/synthetic_${ENV}_${SLURM_JOB_ID}.log" 2>&1 &
done

wait
echo "All synthetic data generated: $(date)"

echo ""
echo "=========================================="
echo "  Diffusion phase complete: $(date)"
echo "  Checkpoints at: $DISA/checkpoints/"
echo "  Synthetic data: $DISA/data/synthetic/"
echo "=========================================="

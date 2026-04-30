#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name="disa_iql"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/iql_%j.out"
#SBATCH --error="logs/iql_%j.err"

echo "=========================================="
echo "  DiSA-RL — Offline IQL Training"
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

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
SEEDS=(0 1 2 3 4)

# 4 GPUs — one env per GPU, all 5 seeds in parallel per GPU
# IQL uses ~500MB VRAM per seed → 5 seeds = ~2.5GB per H200 (141GB) — plenty
for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"

    SYN="./data/synthetic/$ENV/synthetic_transitions.npz"
    if [[ -f "$SYN" ]]; then
        MODE="augmented"
    else
        MODE="offline_only"
        echo "WARNING: No synthetic data for $ENV — using offline_only"
    fi

    echo "GPU $i → $ENV ($MODE) seeds 0-4"

    for seed in "${SEEDS[@]}"; do
        CUDA_VISIBLE_DEVICES=$i python iql/train_iql.py \
            --env       "$ENV" \
            --mode      "$MODE" \
            --seed      "$seed" \
            --num_steps 1000000 \
            --wandb_project disa-rl \
            >> "logs/slurm/iql_${ENV}_s${seed}_${SLURM_JOB_ID}.log" 2>&1 &
    done
done

echo "All IQL jobs launched. Waiting..."
wait
echo "All IQL training complete: $(date)"

echo ""
echo "=========================================="
echo "  IQL phase complete: $(date)"
echo "  Checkpoints at: $DISA/checkpoints/"
echo "=========================================="
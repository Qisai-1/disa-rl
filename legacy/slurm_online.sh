#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem=120G
#SBATCH --job-name="disa_online"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/online_%j.out"
#SBATCH --error="logs/online_%j.err"

echo "=========================================="
echo "  DiSA-RL — Online SAC Training"
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

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    IQL_CKPT="./checkpoints/$ENV/iql/augmented/best.pt"
    DIFF_CKPT="./checkpoints/$ENV/diffusion/offline_final.pt"
    SYN="./data/synthetic/$ENV/synthetic_transitions.npz"

    if [[ ! -f "$IQL_CKPT" ]]; then
        echo "WARNING: No IQL checkpoint for $ENV — skipping"
        continue
    fi

    echo "GPU $i → $ENV seeds 0-4"

    for seed in "${SEEDS[@]}"; do
        CUDA_VISIBLE_DEVICES=$i WANDB_MODE=offline python online_rl/train_online.py \
            --env            "$ENV" \
            --iql_ckpt       "$IQL_CKPT" \
            --diffusion_ckpt "$DIFF_CKPT" \
            --synthetic_data "$SYN" \
            --num_steps      500000 \
            --seed           "$seed" \
            >> "logs/slurm/online_${ENV}_s${seed}_${SLURM_JOB_ID}.log" 2>&1 &
    done
done

echo "All online jobs launched. Waiting..."
wait
echo "All online training complete: $(date)"

echo ""
echo "=========================================="
echo "  Online phase complete: $(date)"
echo "  Checkpoints at: $DISA/checkpoints/*/online/"
echo "=========================================="

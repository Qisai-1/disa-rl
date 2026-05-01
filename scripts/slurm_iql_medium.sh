#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem=64G
#SBATCH --job-name="disa_iql_medium"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/iql_medium_%j.out"
#SBATCH --error="logs/iql_medium_%j.err"

echo "=========================================="
echo "  DiSA-RL — IQL medium-v2 (all alphas)"
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

# ── Configuration ─────────────────────────────────────────────────────────────
# Alpha conditions (no offline_only — run separately as baseline)
#   0.5  → 50% real + 50% synthetic
#   0.25 → 25% real + 75% synthetic
#   0.0  → 100% synthetic (pure diffusion)
ALPHAS=("0.5" "0.25" "0.0")

SEEDS=(0 1 2 3 4)
BC_WEIGHT=0.1

# 4 envs split across 2 GPUs — each GPU handles 2 envs sequentially
GPU_ENVS=(
    "0:halfcheetah-medium-v2"
    "0:walker2d-medium-v2"
    "1:hopper-medium-v2"
    "1:ant-medium-v2"
)

# ── Launch ────────────────────────────────────────────────────────────────────
for GPU in 0 1; do
    (
        for GPU_ENV in "${GPU_ENVS[@]}"; do
            ASSIGNED_GPU="${GPU_ENV%%:*}"
            ENV="${GPU_ENV##*:}"
            [[ "$ASSIGNED_GPU" != "$GPU" ]] && continue

            SYN="./data/synthetic/$ENV/synthetic_transitions.npz"
            if [[ ! -f "$SYN" ]]; then
                echo "GPU $GPU [$ENV] ERROR: No synthetic data at $SYN — skipping"
                continue
            fi

            for ALPHA in "${ALPHAS[@]}"; do
                echo "GPU $GPU [$ENV] alpha=$ALPHA bc=$BC_WEIGHT seeds=0-4  ($(date))"

                for seed in "${SEEDS[@]}"; do
                    CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
                        --env           "$ENV" \
                        --mode          augmented \
                        --alpha         "$ALPHA" \
                        --bc_weight     "$BC_WEIGHT" \
                        --seed          "$seed" \
                        --num_steps     1000000 \
                        --wandb_project disa-rl-medium \
                        >> "logs/slurm/iql_${ENV}_alpha${ALPHA}_s${seed}_${SLURM_JOB_ID}.log" 2>&1 &
                done
                wait
                echo "GPU $GPU [$ENV] Done: alpha=$ALPHA  ($(date))"
            done
        done
    ) &
done

wait
echo ""
echo "=========================================="
echo "  medium-v2 IQL complete: $(date)"
echo "=========================================="
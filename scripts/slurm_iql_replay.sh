#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem=64G
#SBATCH --job-name="disa_iql_replay"
#SBATCH --account=mech-ai
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/iql_replay_%j.out"
#SBATCH --error="logs/iql_replay_%j.err"

echo "=========================================="
echo "  DiSA-RL — IQL medium-replay-v2 (all alphas)"
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
ALPHAS=("0.5" )
SEEDS=( 2 3 4)
BC_WEIGHT=0.1

ENVS=(
    "halfcheetah-medium-replay-v2"
    "hopper-medium-replay-v2"
    "walker2d-medium-replay-v2"
    "ant-medium-replay-v2"
)

# ── Launch — 4 GPUs, one env per GPU ─────────────────────────────────────────
for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU=$i
    SYN="./data/synthetic/$ENV/synthetic_transitions.npz"

    (
        if [[ ! -f "$SYN" ]]; then
            echo "GPU $GPU [$ENV] ERROR: No synthetic data at $SYN — skipping"
            exit 0
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
                    --expectile     0.7 \
                    --temperature   3.0 \
                    --wandb_project disa-rl-replay \
                    >> "logs/slurm/iql_${ENV}_alpha${ALPHA}_s${seed}_${SLURM_JOB_ID}.log" 2>&1 &
            done
            wait
            echo "GPU $GPU [$ENV] Done: alpha=$ALPHA  ($(date))"
        done
    ) &
done

wait
echo ""
echo "=========================================="
echo "  medium-replay-v2 IQL complete: $(date)"
echo "=========================================="
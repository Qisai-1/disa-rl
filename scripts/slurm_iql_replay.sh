#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem=120G
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

ENVS=(
    "halfcheetah-medium-replay-v2"
    "hopper-medium-replay-v2"
    "walker2d-medium-replay-v2"
    "ant-medium-replay-v2"
)
SEEDS=(0 1 2 3 4)
ALPHAS=("1.0:offline_only" "0.75:augmented" "0.5:augmented" "0.0:augmented")

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU=$i
    SYN="./data/synthetic/$ENV/synthetic_transitions.npz"

    (
        for ALPHA_MODE in "${ALPHAS[@]}"; do
            ALPHA="${ALPHA_MODE%%:*}"
            MODE="${ALPHA_MODE##*:}"

            if [[ "$MODE" == "augmented" ]] && [[ ! -f "$SYN" ]]; then
                echo "GPU $GPU [$ENV] WARNING: No synthetic data — skipping alpha=$ALPHA"
                continue
            fi

            echo "GPU $GPU [$ENV] mode=$MODE alpha=$ALPHA seeds=0-4  ($(date))"

            for seed in "${SEEDS[@]}"; do
                CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
                    --env           "$ENV" \
                    --mode          "$MODE" \
                    --alpha         "$ALPHA" \
                    --seed          "$seed" \
                    --num_steps     1000000 \
                    --wandb_project disa-rl \
                    >> "logs/slurm/iql_${ENV}_${MODE}_alpha${ALPHA}_s${seed}_${SLURM_JOB_ID}.log" 2>&1 &
            done
            wait
            echo "GPU $GPU [$ENV] Done: mode=$MODE alpha=$ALPHA  ($(date))"
        done
    ) &
done

wait
echo ""
echo "=========================================="
echo "  medium-replay-v2 IQL complete: $(date)"
echo "=========================================="
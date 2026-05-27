#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Phase B — Regenerate synthetic data for the 4 medium-replay envs.
#
#  Run after Phase A (retrain_diffusion_replay.sh) finishes. Uses the freshly
#  retrained diffusion models in checkpoints/<env>/diffusion/offline_final.pt.
#  Conditioning fix already applied: generate_synthetic_data.py now defaults to
#  --return_sampling real (was "topk") — samples target_return from the full
#  sub-traj return distribution, not only the top-20% band.
#
#  USAGE:
#    bash scripts/regen_syn_data_replay.sh
#
#  Runs 4 in parallel on the 4 GPUs. Each env generates 1M transitions.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

# ── Preflight ───────────────────────────────────────────────────────────────
if ! timeout 60 python -c "import torch; assert torch.cuda.is_available(); \
        torch.randn(8,device='cuda').sum().item()" 2>/dev/null; then
    echo "FATAL: CUDA not usable on $(hostname)."
    exit 1
fi
N_GPU=$(python -c 'import torch; print(torch.cuda.device_count())')
[ "$N_GPU" -lt 4 ] && { echo "ERROR: need 4 GPUs, have $N_GPU"; exit 1; }
echo "preflight: CUDA OK — $N_GPU GPU(s) visible"

# Refuse to launch if Phase A diffusion training still running.
if pgrep -f 'diffusion/train.py' > /dev/null; then
    echo "ERROR: diffusion/train.py still running — Phase A not done."
    echo "Wait for Phase A to finish before launching Phase B."
    exit 1
fi
# Refuse to double-launch regeneration.
if pgrep -f 'generate_synthetic_data.py' > /dev/null; then
    echo "ERROR: generate_synthetic_data.py already running."
    exit 1
fi

# Sanity: confirm freshly-trained checkpoints exist.
ENVS=(
    halfcheetah-medium-replay-v2
    hopper-medium-replay-v2
    walker2d-medium-replay-v2
    ant-medium-replay-v2
)
for ENV in "${ENVS[@]}"; do
    CKPT="checkpoints/$ENV/diffusion/offline_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo "FATAL: missing $CKPT — Phase A did not complete for $ENV"
        exit 1
    fi
done

# Archive the old syn data (the ones audited as broken/biased), keep for reference.
TS=$(date +%Y%m%d_%H%M%S)
for ENV in "${ENVS[@]}"; do
    OLD="data/synthetic/$ENV"
    if [ -d "$OLD" ]; then
        mv "$OLD" "${OLD}_pre_retrain_$TS"
        echo "archived  $OLD  →  ${OLD}_pre_retrain_$TS"
    fi
done

unset WANDB_MODE
mkdir -p logs

echo
echo "=== Phase B regen launching at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u generate_synthetic_data.py \
        --env "$ENV" \
        --n_transitions 1000000 \
        --batch_size 64 \
        --return_sampling real \
        > "logs/regen_${ENV}.log" 2>&1 &
    echo "  GPU $i  →  $ENV   (logs/regen_${ENV}.log)"
done

sleep 30
n=$(pgrep -fc 'generate_synthetic_data.py')
echo
echo "=== launched $n generate_synthetic_data.py processes (target: 4) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

Phase B regen started.

  - Watch a log:      tail -F logs/regen_halfcheetah-medium-replay-v2.log
  - Done when:        ls data/synthetic/*-medium-replay-v2/synthetic_transitions.npz | wc -l == 4
  - Kill everything:  pkill -f 'generate_synthetic_data.py'

  After all four finish, Phase C: re-run
    python scripts/validate_synthetic_data.py
  to see whether the audit clears.

EOM

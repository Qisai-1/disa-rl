#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Retrain the 4 medium-replay diffusion models — in parallel on the 4 GPUs.
#
#  WHY:
#    The originals stopped at ~40-50k steps (patience=20 early-stopping fired
#    fast on the small replay datasets). Per [[project-syn-data-audit]], that
#    left the 4 replay models undertrained — hc-medium-replay produced a near-
#    constant reward and absurd velocity (mean 12.3); the others compressed
#    reward variance to 0.4-0.5× real. Retraining with patience=200 forces
#    much longer training before stopping.
#
#  HOW:
#    One model per GPU, all 4 in parallel. Each writes to its env's diffusion
#    checkpoint dir; the old undertrained `offline_*.pt` are archived to
#    `checkpoints/<env>/diffusion_undertrained_<timestamp>/` first so they're
#    preserved for comparison rather than overwritten.
#
#  USAGE:
#    bash scripts/retrain_diffusion_replay.sh
#
#  Run inside the salloc shell on a 4-GPU node, in tmux. ~6-12h wall-clock.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

# ── Preflight: CUDA must actually be usable ────────────────────────────────
if ! timeout 60 python -c "import torch; assert torch.cuda.is_available(); \
        torch.randn(8,device='cuda').sum().item()" 2>/dev/null; then
    echo "FATAL: CUDA is not usable on this node ($(hostname))."
    echo "  Check nvidia-fabricmanager / nvidia-smi and retry."
    exit 1
fi
N_GPU=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "preflight: CUDA OK — $N_GPU GPU(s) visible"
[ "$N_GPU" -lt 4 ] && { echo "ERROR: need 4 GPUs, have $N_GPU"; exit 1; }

if [ -z "$TMUX" ]; then
    echo "WARNING: this shell is NOT inside tmux."
    echo "  If SSH disconnects, the salloc dies and all 4 jobs are killed."
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# Refuse to re-launch if diffusion training is already running.
if pgrep -f 'diffusion/train.py' > /dev/null; then
    echo "ERROR: diffusion/train.py already running — refusing to launch."
    echo "Kill first:  pkill -f 'diffusion/train.py'"
    exit 1
fi

unset WANDB_MODE   # online; set WANDB_MODE=offline before running if init hangs

ENVS=(
    halfcheetah-medium-replay-v2
    hopper-medium-replay-v2
    walker2d-medium-replay-v2
    ant-medium-replay-v2
)

# Archive the undertrained checkpoints so they're preserved, not overwritten.
TS=$(date +%Y%m%d_%H%M%S)
for ENV in "${ENVS[@]}"; do
    D="checkpoints/$ENV/diffusion"
    if [ -d "$D" ]; then
        mv "$D" "${D}_undertrained_$TS"
        echo "archived  $D  →  ${D}_undertrained_$TS"
    fi
done

mkdir -p logs

echo
echo "=== diffusion retrain launching at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u diffusion/train.py \
        --env        "$ENV" \
        --batch_size 256 \
        --lr         1e-4 \
        --patience   200 \
        --num_steps  300000 \
        > "logs/diffusion_retrain_${ENV}.log" 2>&1 &
    echo "  GPU $i  →  $ENV   (logs/diffusion_retrain_${ENV}.log)"
done

sleep 30
n=$(pgrep -fc 'diffusion/train.py')
echo
echo "=== launched $n diffusion/train.py processes (target: 4) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

Diffusion retrain (replay envs) started.

  - Detach tmux with  Ctrl-b d   (KEEP the salloc alive — 4 jobs depend on it)
  - Watch a log:      tail -F logs/diffusion_retrain_halfcheetah-medium-replay-v2.log
  - Quick progress:   for f in logs/diffusion_retrain_*.log; do
                        echo "=== $f ==="; tail -1 "$f"; done
  - Kill everything:  pkill -f 'diffusion/train.py'

  Each model targets 300k steps with patience=200. If the loss truly plateaus
  the early-stop will still fire; we'll see how far each gets from the logs.
  After all four finish, regenerate syn data and re-run
    python scripts/validate_synthetic_data.py
  to see whether the audit clears.

EOM

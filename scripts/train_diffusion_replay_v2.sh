#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Phase B — Train v2 diffusion (with reward) on the 4 medium-replay envs.
#  Bundled "best foundation" build per user sign-off 2026-05-26.
#
#  Changes vs v1 (Phase A):
#    - Reward-in-diffusion (D = obs+action+1; per-channel z-score in data_v2)
#    - Bigger model: hidden 256→512, depth 6→8, heads 4→8 (~5M → ~39M params)
#    - Lower dropout: 0.3 → 0.15
#    - Gaussian data aug σ=0.01
#    - λ_reward = 1.0 (equal weight on reward velocity loss)
#
#  Output: checkpoints/<env>/diffusion_v2/ (separate from v1's diffusion/)
#
#  USAGE:
#    bash scripts/train_diffusion_replay_v2.sh
#
#  ~13h wall-clock on 4 A40s (2× v1 due to bigger model).
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

if ! timeout 60 python -c "import torch; assert torch.cuda.is_available(); \
        torch.randn(8,device='cuda').sum().item()" 2>/dev/null; then
    echo "FATAL: CUDA not usable on $(hostname)."
    exit 1
fi
N_GPU=$(python -c 'import torch; print(torch.cuda.device_count())')
[ "$N_GPU" -lt 4 ] && { echo "ERROR: need 4 GPUs, have $N_GPU"; exit 1; }
echo "preflight: CUDA OK — $N_GPU GPU(s) visible"

# Refuse to launch if v1 retrain (Phase A) still running on the same GPUs.
# Match only actual python processes (not shells that happen to have the path
# in their cmdline — e.g. monitoring grep commands).
if pgrep -f 'python.*diffusion/train\.py' > /dev/null; then
    echo "ERROR: v1 diffusion/train.py still running — Phase A not done."
    echo "Wait for Phase A to finish (or pkill it) before launching v2."
    exit 1
fi
# Refuse to double-launch v2.
if pgrep -f 'python.*diffusion/train_v2\.py' > /dev/null; then
    echo "ERROR: diffusion/train_v2.py already running."
    exit 1
fi

ENVS=(
    halfcheetah-medium-replay-v2
    hopper-medium-replay-v2
    walker2d-medium-replay-v2
    ant-medium-replay-v2
)

# Archive any existing v2 checkpoint dirs (defensive; usually none)
TS=$(date +%Y%m%d_%H%M%S)
for ENV in "${ENVS[@]}"; do
    D="checkpoints/$ENV/diffusion_v2"
    if [ -d "$D" ]; then
        mv "$D" "${D}_old_$TS"
        echo "archived  $D  →  ${D}_old_$TS"
    fi
done

unset WANDB_MODE
mkdir -p logs

echo
echo "=== Phase B v2 retrain launching at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u diffusion/train_v2.py \
        --env "$ENV" \
        --hidden_size 512 --depth 8 --num_heads 8 \
        --mlp_dropout 0.15 --train_noise 0.01 --lambda_reward 1.0 \
        --batch_size 256 --lr 1e-4 \
        --patience 200 --num_steps 300000 \
        --wandb_project disa-rl-v2 \
        > "logs/diffusion_v2_${ENV}.log" 2>&1 &
    echo "  GPU $i  →  $ENV   (logs/diffusion_v2_${ENV}.log)"
done

sleep 30
n=$(pgrep -fc 'diffusion/train_v2.py')
echo
echo "=== launched $n diffusion/train_v2.py processes (target: 4) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

Phase B v2 retrain started.

  - Watch a log:      tail -F logs/diffusion_v2_halfcheetah-medium-replay-v2.log
  - Done when:        grep -l "offline_final.pt" logs/diffusion_v2_*.log | wc -l == 4
  - Kill everything:  pkill -f 'diffusion/train_v2.py'

  Each model targets 300k steps with patience=200. Bigger model ⇒ ~13h wall-clock.
  After all 4 finish, regenerate syn data with both v1 and v2 architectures
  and run the audit comparison.

EOM

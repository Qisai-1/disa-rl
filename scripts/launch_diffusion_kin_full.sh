#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Full retrain of the kinematic-consistency generators for the two well-posed
#  envs (hopper, walker2d). HalfCheetah is excluded (Euler ill-posed at dt=0.05,
#  and it's already clean downstream). Ant handled separately (quaternion).
#
#  Fills 4 GPUs: {hopper, walker2d} × two λ_dyn weights, 300k steps each,
#  writing to checkpoints/<env>/diffusion_kin_w<weight>/  (never touches the
#  production diffusion/ dir).
#
#  USAGE:
#     bash scripts/launch_diffusion_kin_full.sh <W1> <W2>
#  e.g. bash scripts/launch_diffusion_kin_full.sh 0.1 0.3
#  W1/W2 = the two best λ_dyn from the hopper validation sweep.
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

W1="${1:?need weight 1, e.g. 0.1}"
W2="${2:?need weight 2, e.g. 0.3}"

if ! timeout 60 python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FATAL: CUDA not usable on $(hostname)."; exit 1
fi
if pgrep -f 'python -u diffusion/train.py' >/dev/null; then
    echo "ERROR: diffusion/train.py already running — refusing to launch."; exit 1
fi

mkdir -p logs
sub() { echo "diffusion_kin_w$(echo $1 | tr -d '.')"; }   # 0.1 -> diffusion_kin_w01

echo "=== full kinematic retrain launching $(date '+%F %T')  (W1=$W1 W2=$W2) ==="
i=0
for ENV in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for W in "$W1" "$W2"; do
    S=$(sub "$W")
    CUDA_VISIBLE_DEVICES=$i python -u diffusion/train.py \
        --env "$ENV" --dyn_weight "$W" --output_subdir "$S" \
        --num_steps 300000 --batch_size 256 --patience 200 \
        > "logs/kinfull_${ENV}_${S}.log" 2>&1 &
    echo "  GPU $i → $ENV  dyn=$W → $S"
    i=$((i+1))
  done
done

until [ "$(pgrep -fc 'python -u diffusion/train.py')" -ge 4 ]; do sleep 2; done
echo "launched $(pgrep -fc 'python -u diffusion/train.py') runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

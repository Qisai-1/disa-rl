#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Diffusion ensemble training — K seeds for the "calibrated stitching"
#  thesis. At generation time we'll run all K models, sample one trajectory
#  per condition from each, and FILTER by pairwise disagreement (high agreement
#  = trustworthy syn; high disagreement = drop).
#
#  Why K=3: 3 is the minimum that gives a non-trivial "pairwise variance"
#  signal. Each additional seed = +1 full diffusion training run (≈4h on
#  H200), so 3 keeps the compute reasonable while giving 3 pairs (0-1, 0-2,
#  1-2) for the disagreement filter.
#
#  Architecture: identical to production (hidden=256, depth=6, 300k steps).
#  Only the seed and the output_subdir differ.
#
#  Outputs:
#    checkpoints/<env>/diffusion_ens_s{0,1,2}/best.pt + offline_final.pt
#    logs/diffens_<env>_s<SEED>.log
#
#  GPU layout: K * 2 envs = 6 runs. With 3 H200s, that's 2/GPU — comfortable.
#  ETA ~6h on H200 (diffusion is compute-bound, single-stream).
#
#  USAGE:
#    NGPU=3 bash scripts/launch_diffusion_ensemble.sh
#    (Wait for Stage 2.5 to finish first — diffusion saturates a GPU.)
#
#  NB: seed 0 will RE-CREATE the production diffusion model if you don't
#  guard. We use diffusion_ens_s0 instead of overwriting diffusion/.
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

NGPU=${NGPU:-4}
K=${K:-3}

if pgrep -f 'python -u (iql/train_iql|diffusion/train).py' >/dev/null; then
    echo "ERROR: training procs already running. Wait."; exit 1
fi

mkdir -p logs

i=0
run() {  # env seed
  local env=$1 seed=$2 gpu=$(( i % NGPU )); i=$((i+1))
  local tag="${env%%-*}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python -u diffusion/train.py \
      --env "$env" --seed "$seed" \
      --batch_size 256 --lr 1e-4 --patience 200 --num_steps 300000 \
      --output_subdir "diffusion_ens_s${seed}" \
      > "logs/diffens_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  seed=$seed  → checkpoints/${env}/diffusion_ens_s${seed}/"
}

echo "=== Diffusion ensemble launching $(date '+%F %T') ==="
echo "    K=${K} seeds × {hopper, walker}"
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for seed in $(seq 0 $((K-1))); do
    run "$env" "$seed"
  done
done

until [ "$(pgrep -fc 'diffusion/train.py')" -ge $((K * 2)) ]; do sleep 3; done
echo "launched $(pgrep -fc 'diffusion/train.py') / $((K * 2)) runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

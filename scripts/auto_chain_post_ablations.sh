#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain #4: post-ablations → diffusion ensemble training.
#
#  Fires when the leave-one-out ablations (wave1) complete. Launches the
#  diffusion ensemble training (6 runs × ~6h) on the freed GPUs.
#
#  USAGE:
#    NGPU=3 nohup bash scripts/auto_chain_post_ablations.sh > logs/auto_chain_4.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain-4 armed; waiting for ablations to launch + finish..."

while [ "$(pgrep -fc 'wandb_project disa-rl-s2p7-abl')" -eq 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] Ablations detected; now waiting for completion..."

while [ "$(pgrep -fc 'wandb_project disa-rl-s2p7-abl')" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] Ablations done. Aggregating..."

python scripts/report_runs.py 'logs/s2p7abl_*.log' > results/stage2p7_ablations.txt 2>&1 || true
cat results/stage2p7_ablations.txt | tail -20

# Sanity check before launching diffusion
if pgrep -f 'python -u (iql|diffusion)/train.py' >/dev/null; then
    echo "[$(date '+%F %T')] Training procs still alive — abort"
    pgrep -fa 'python -u' | head -3
    exit 1
fi

echo "[$(date '+%F %T')] Launching diffusion ensemble training (6 runs × ~6h)..."
NGPU=$NGPU bash scripts/launch_diffusion_ensemble.sh
echo "[$(date '+%F %T')] === Auto-chain-4 complete; diffusion ensemble training ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

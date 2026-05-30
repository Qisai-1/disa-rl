#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain TD3+BC sweep as soon as Stage 2.6 AND 2.7 procs both exit.
#
#  Second link in the autonomous run. Run AFTER auto_chain_post_s2p5.sh.
#
#  USAGE:
#    NGPU=3 nohup bash scripts/auto_chain_post_s2p7.sh > logs/auto_chain_2.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain-2 armed; waiting for Stage 2.6 and 2.7 to land..."

# First wait for them to LAUNCH (in case this is started before auto-chain-1 fires)
while [ "$(pgrep -fc 'wandb_project disa-rl-s2p6')" -eq 0 ] && \
      [ "$(pgrep -fc 'wandb_project disa-rl-s2p7')" -eq 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] 2.6/2.7 procs detected, now waiting for them to finish..."

# Then wait for BOTH to finish
while [ "$(pgrep -fc 'wandb_project disa-rl-s2p6')" -gt 0 ] || \
      [ "$(pgrep -fc 'wandb_project disa-rl-s2p7')" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] 2.6 and 2.7 both exited. Aggregating..."

python scripts/report_runs.py 'logs/s2p6_*.log' > results/stage2p6_final.txt 2>&1 || true
python scripts/report_runs.py 'logs/s2p7_*.log' > results/stage2p7_final.txt 2>&1 || true
echo "--- Stage 2.6 ---"
cat results/stage2p6_final.txt | tail -10
echo "--- Stage 2.7 ---"
cat results/stage2p7_final.txt | tail -10

# Sanity check: GPUs should be idle
if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "[$(date '+%F %T')] Some IQL procs still alive (not 2.6/2.7) — abort"
    pgrep -fa 'python -u iql/train_iql.py' | head -3
    exit 1
fi

echo "[$(date '+%F %T')] Launching TD3+BC sweep (18 runs)..."
NGPU=$NGPU bash scripts/launch_td3bc_sweep.sh
echo "[$(date '+%F %T')] TD3+BC sweep launched"

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "[$(date '+%F %T')] === Auto-chain-2 complete; TD3+BC training ==="

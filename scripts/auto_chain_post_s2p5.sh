#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain Stage 2.6 + 2.7 launchers as soon as Stage 2.5 procs exit.
#
#  Used by the autonomous-run plan: Claude doesn't need to be triggered
#  by an external signal; this script blocks on Stage 2.5 completion,
#  aggregates the final numbers, and fires 2.6+2.7 concurrently.
#
#  USAGE (run in background or via Monitor):
#    NGPU=3 nohup bash scripts/auto_chain_post_s2p5.sh > logs/auto_chain.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain armed; waiting for Stage 2.5 to finish..."

# Wait for Stage 2.5 procs to fully exit
while pgrep -fc 'wandb_project disa-rl-s2p5' >/dev/null && \
      [ "$(pgrep -fc 'wandb_project disa-rl-s2p5')" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] Stage 2.5 procs all exited. Aggregating final numbers..."

# Save final Stage 2.5 snapshot
python scripts/report_runs.py 'logs/s2p5_*.log' > results/stage2p5_final.txt 2>&1 || true
cat results/stage2p5_final.txt | tail -20

# Sanity: refuse to chain if Stage 2.5 procs are still alive (race condition)
if pgrep -fc 'wandb_project disa-rl-s2p5' >/dev/null && \
   [ "$(pgrep -fc 'wandb_project disa-rl-s2p5')" -gt 0 ]; then
    echo "[$(date '+%F %T')] STAGE 2.5 STILL ALIVE — abort chain to avoid GPU oversubscription"
    exit 1
fi

echo "[$(date '+%F %T')] Launching Stage 2.6 (CAPA+GTA, --no-syn_normalize_rewards)..."
NGPU=$NGPU bash scripts/launch_stage2p6_keepamp.sh 0.5 0.25
echo "[$(date '+%F %T')] Stage 2.6 launched (6 runs)"

# Give 2.6 procs ~60s to claim CUDA contexts before 2.7 lands
sleep 60

echo "[$(date '+%F %T')] Launching Stage 2.7 (full novelty pack) with SKIP_GUARD=1..."
SKIP_GUARD=1 NGPU=$NGPU bash scripts/launch_stage2p7_bestcombo.sh
echo "[$(date '+%F %T')] Stage 2.7 launched (6 runs)"

echo "[$(date '+%F %T')] === Auto-chain complete; 12 runs (s2p6 + s2p7) now training ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

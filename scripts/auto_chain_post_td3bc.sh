#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain #3: post-TD3+BC verdict gate.
#
#  Triggers when TD3+BC sweep finishes. Reads the Stage 2.7 verdict via
#  scripts/verdict_check.py. If WIN or MIXED: launches the diffusion
#  ensemble training (6 runs × ~6h) AND the leave-one-out ablations
#  (12 runs wave1, ~5h) — staggered so they don't clobber each other.
#
#  If LOSS: writes a debug note to results/post_td3bc_loss.txt and exits.
#
#  USAGE:
#    NGPU=3 nohup bash scripts/auto_chain_post_td3bc.sh > logs/auto_chain_3.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain-3 armed; waiting for TD3+BC to land + finish..."

# First wait for TD3+BC to LAUNCH
while [ "$(pgrep -fc 'wandb_project disa-rl-td3bc')" -eq 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] TD3+BC procs detected; now waiting for completion..."

while [ "$(pgrep -fc 'wandb_project disa-rl-td3bc')" -gt 0 ]; do
    sleep 60
done
echo "[$(date '+%F %T')] TD3+BC done. Aggregating..."

python scripts/report_runs.py 'logs/td3bc_*.log' > results/td3bc_final.txt 2>&1 || true
cat results/td3bc_final.txt | tail -15

# Check the Stage 2.7 verdict
echo "[$(date '+%F %T')] Checking Stage 2.7 verdict..."
python scripts/verdict_check.py 2>&1 | tee results/stage2p7_verdict.txt
VERDICT=$(grep -oE 'VERDICT=[A-Z]+' results/stage2p7_verdict.txt | tail -1 | cut -d= -f2)
echo "[$(date '+%F %T')] VERDICT: $VERDICT"

# Sanity: refuse to launch if anything still alive
if pgrep -f 'python -u (iql|diffusion)/train.py' >/dev/null; then
    echo "[$(date '+%F %T')] Training procs still alive — abort to avoid GPU oversubscription"
    pgrep -fa 'python -u' | head -5
    exit 1
fi

case $VERDICT in
  WIN|MIXED)
    echo "[$(date '+%F %T')] Verdict $VERDICT → launching diffusion ensemble (background) + ablations (foreground)..."
    # Ablations on the IQL backbone — 12 runs at 4/GPU, ~5h
    NGPU=$NGPU bash scripts/launch_stage2p7_ablations.sh wave1
    echo "[$(date '+%F %T')] Ablations wave1 launched"
    # Don't launch diffusion ensemble concurrently — would saturate GPUs.
    # Queue a follow-up auto-chain to fire it AFTER ablations.
    NGPU=$NGPU nohup bash scripts/auto_chain_post_ablations.sh \
        > logs/auto_chain_4.log 2>&1 &
    echo "[$(date '+%F %T')] Auto-chain-4 (post-ablations → diffusion ensemble) armed (PID $!)"
    ;;
  LOSS)
    echo "[$(date '+%F %T')] Verdict LOSS — not launching ablations / ensemble."
    cat > results/post_td3bc_loss.txt <<EOF
Stage 2.7 verdict: LOSS — novelty pack did not beat the lifted Stage 2.5 baseline.
Investigate before launching more. Possible causes:
  - Novelty pack hyperparameters wrong (gbc_weight 0.05 too low, etc.)
  - Curriculum warmup 50000 too short; novelty pack still warming when stopped
  - CAPA+GTA fundamental mismatch with strong offline baseline → need different angle
Recommended next step: examine train/critic_syn_coef_live, train/n_gbc, train/uncert_gate_syn_mean
in wandb to see which novelty pieces were even active.
EOF
    cat results/post_td3bc_loss.txt
    ;;
  *)
    echo "[$(date '+%F %T')] Unknown verdict '$VERDICT' — abort"
    exit 1
    ;;
esac

echo "[$(date '+%F %T')] === Auto-chain-3 complete ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

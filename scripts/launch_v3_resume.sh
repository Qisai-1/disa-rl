#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  DiSA-RL v3 sweep — RESUME the killed 2026-05-18 run
#
#  The 2026-05-18 v3 sweep was killed when the allocation expired at ~16:05.
#  It reached step ~100k-200k of the 500k target. Every run saved a
#  step_*.pt checkpoint (full optimizer + step state) every 100k steps, so
#  train_iql.py --resume picks up the latest one and continues to 500k.
#
#  Run on the GPU node inside the salloc shell:
#      bash scripts/launch_v3_resume.sh
#
#  Re-launches the same 12 jobs across 4 A100s (3 per GPU), each with the
#  EXACT same hyperparameters as the original sweep (required — the agent
#  architecture must match the checkpoint) plus --resume.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

# ── 0. Environment ─────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

# ── Preflight: CUDA must actually be usable on this node ───────────────────
# A failed nvidia-fabricmanager (seen on nova22-amp-7, 2026-05-21) leaves the
# A100 SXM GPUs visible to nvidia-smi but DEAD for CUDA — every run then dies
# with "Error 802: system not yet initialized". Abort loudly before launching.
if ! timeout 60 python -c "import torch; assert torch.cuda.is_available(); torch.randn(8,device='cuda').sum().item()" 2>/dev/null; then
    echo "FATAL: CUDA is not usable on this node ($(hostname))."
    echo "  Likely a failed nvidia-fabricmanager. Check:"
    echo "    nvidia-smi -q | grep -A2 '^    Fabric'      (State should be 'Completed')"
    echo "    systemctl is-active nvidia-fabricmanager     (should be 'active')"
    echo "  Get a different GPU node (salloc --exclude=<bad-node>) and retry."
    exit 1
fi
echo "preflight: CUDA OK — $(python -c 'import torch; print(torch.cuda.device_count())') GPU(s) visible"

if [ -z "$TMUX" ]; then
    echo "WARNING: this shell is NOT inside tmux."
    echo "  If SSH disconnects, the salloc dies and all 12 jobs are killed again."
    echo "  Suggested: re-run inside  'tmux new -s salloc'  before launching."
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

unset WANDB_MODE   # online; set WANDB_MODE=offline before running if init hangs

# Archive the killed-run logs (step 1-~200k history) so the resumed runs,
# which start a fresh log, don't clobber them.
ARCHIVE="logs/old/v3_killed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE"
mv logs/v3_*.log "$ARCHIVE"/ 2>/dev/null || true
echo "archived killed-run logs to $ARCHIVE/"

# Refuse to re-launch if training is already running.
n_running=$(pgrep -f 'iql/train_iql.py' | wc -l)
if [ "$n_running" -gt 0 ]; then
    echo "ERROR: $n_running 'train_iql.py' processes already running — refusing to launch."
    echo "Kill them first:  pkill -f 'iql/train_iql.py'"
    exit 1
fi

# ── 1. Per-env config (must match the original v3 sweep exactly) ───────────
ENVS_MEDIUM=( halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 )
ENVS_REPLAY=( halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2 ant-medium-replay-v2 )

reward_scale_for() {
    case "$1" in
        halfcheetah*|ant*) echo 5  ;;
        hopper*|walker2d*) echo 10 ;;
        *)                 echo 1  ;;
    esac
}

critic_subset_for() {
    case "$1" in
        hopper*|walker2d*) echo 10 ;;
        *)                 echo 2  ;;
    esac
}

# ── 2. Re-launch the 12 jobs with --resume ─────────────────────────────────
echo "=== v3 RESUME launching at $(date '+%F %T') ==="

# Baselines (real-only IQL)
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode offline_only --bc_weight 0.0 --seed 0 \
        --num_steps 500000 --resume --save_every 25000 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --action_noise_std 0.05 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-v3-baseline \
        > "logs/v3_baseline_${ENV}_s0.log" 2>&1 &
done

# DRC-IQL on medium-v2
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.5 --seed 0 \
        --num_steps 500000 --resume --save_every 25000 --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --sa_iql --expectile_real 0.9 --expectile_syn 0.5 --sa_clip 0.5 2.0 \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-v3-drc \
        > "logs/v3_drc_${ENV}_s0.log" 2>&1 &
done

# DRC-IQL on medium-replay-v2
for i in 0 1 2 3; do
    ENV=${ENVS_REPLAY[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.5 --seed 0 \
        --num_steps 500000 --resume --save_every 25000 --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --sa_iql --expectile_real 0.9 --expectile_syn 0.5 --sa_clip 0.5 2.0 \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-v3-drc-replay \
        > "logs/v3_drc-replay_${ENV}_s0.log" 2>&1 &
done

sleep 30
n=$(pgrep -f 'iql/train_iql.py' | wc -l)
echo
echo "=== launched $n train_iql.py processes (target: 12) ==="
echo "Each log should print a 'Resuming from step_0XXXXXX.pt' line near the top."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

v3 resume started.

  - Detach tmux with  Ctrl-b d   (KEEP the salloc alive)
  - Confirm resume:   grep -h Resuming logs/v3_*.log
  - Watch a log:      tail -F logs/v3_drc_hopper-medium-v2_s0.log
  - Snapshot scores:  python scripts/aggregate_results.py --logs_glob "logs/v3_*.log"
  - Kill everything:  pkill -f 'iql/train_iql.py'

EOM

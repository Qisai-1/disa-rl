#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  DiSA-RL v3 sweep — full DRC-IQL + PARS + alpha-warmup (with bug fix)
#
#  Run on the GPU node inside the salloc shell:
#      bash scripts/launch_v3_sweep.sh
#
#  Launches 12 jobs across 4 A100s (3 per GPU):
#    • 4 baselines  (IQL real-only on medium-v2)         GPU 0-3
#    • 4 DRC        (DRC-IQL on medium-v2 + α-warmup)    GPU 0-3
#    • 4 DRC-replay (DRC-IQL on medium-replay-v2)        GPU 0-3
#
#  Each runs for 500_000 training steps (~10-12h on A100 with 3/GPU).
#  Plus a watchdog that emails alloydas@iastate.edu when all 12 finish.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

# ── 0. Environment ─────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

if [ -z "$TMUX" ]; then
    echo "⚠️  WARNING: this shell is NOT inside tmux."
    echo "   If your SSH disconnects, the salloc dies and all 12 jobs are killed."
    echo "   Suggested: re-run inside  'tmux new -s salloc'  before launching."
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# WANDB stays in default (online) mode so you can watch on wandb.ai.
# If wandb-init hangs again (we saw 3-min stalls on a fresh node last time),
# you can fall back to offline mode by setting:  export WANDB_MODE=offline
# before re-running this script.
unset WANDB_MODE
mkdir -p logs/old

# Archive any previous v2 / v3 logs so we don't confuse old crash tracebacks
# with the new run.
mv logs/v2_*.log logs/old/ 2>/dev/null || true
mv logs/v3_*.log logs/old/ 2>/dev/null || true

# Refuse to re-launch if some training is already running — avoids
# accidentally double-launching when the script is run twice.
n_running=$(pgrep -f 'iql/train_iql.py' | wc -l)
if [ "$n_running" -gt 0 ]; then
    echo "ERROR: $n_running 'train_iql.py' processes already running — refusing to launch."
    echo "Either let them finish, or kill them first:  pkill -f 'iql/train_iql.py'"
    exit 1
fi

# ── 1. Per-env config (PARS-recommended scales) ────────────────────────────
ENVS_MEDIUM=( halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 )
ENVS_REPLAY=( halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2 ant-medium-replay-v2 )

reward_scale_for() {
    case "$1" in
        halfcheetah*|ant*) echo 5  ;;   # PARS halfcheetah_medium_expert / mujoco offline_to_online
        hopper*|walker2d*) echo 10 ;;   # PARS hopper / walker2d configs
        *)                 echo 1  ;;
    esac
}

critic_subset_for() {
    # PARS hopper config uses num_sample_ensemble=10 (full-min over the 10 critics);
    # halfcheetah / ant use 2 (REDQ-style random pair). Mirror their choices.
    case "$1" in
        hopper*|walker2d*) echo 10 ;;
        *)                 echo 2  ;;
    esac
}

# ── 2. Launch the 12 jobs ──────────────────────────────────────────────────
echo "=== v3 sweep launching at $(date '+%F %T') ==="

# Baselines (real-only IQL with PARS bits)
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode offline_only --bc_weight 0.0 --seed 0 \
        --num_steps 500000 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --action_noise_std 0.05 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-v3-baseline \
        > "logs/v3_baseline_${ENV}_s0.log" 2>&1 &
done

# DRC-IQL on medium-v2 (full novelty stack)
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.5 --seed 0 \
        --num_steps 500000 --bc_weight 0.1 \
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
        --num_steps 500000 --bc_weight 0.1 \
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
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ── 3. Email watchdog ──────────────────────────────────────────────────────
echo
if pgrep -f email_on_done >/dev/null; then
    echo "watchdog already running — skipping re-launch"
else
    nohup bash scripts/email_on_done.sh > logs/watchdog.log 2>&1 &
    disown
    echo "watchdog launched, pid=$!"
fi

# ── 4. Final instructions ──────────────────────────────────────────────────
cat <<'EOM'

✅ v3 sweep started.

Now:
  1. Detach this tmux session with  Ctrl-b d        (KEEP the salloc alive)
  2. Reattach later with             tmux attach -t salloc-main   (or whatever name)
  3. Check progress anytime with     bash scripts/status.sh
  4. The email will arrive when all 12 jobs finish (~10-12 hours)

If anything looks wrong:
  • View any log live:   tail -F logs/v3_*_s0.log
  • Snapshot scores:     python scripts/aggregate_results.py --logs_glob "logs/v3_*.log"
  • Kill everything:     pkill -f 'iql/train_iql.py'

EOM

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  DiSA-RL — Tier-1 ablation (the "make augmented beat baseline" diagnostic)
#
#  WHY: the v3 DRC-IQL config (--alpha 0.5 --sa_iql --expectile_syn 0.5)
#  underperforms the real-only baseline on every env, stable across the
#  2026-05-12 and 2026-05-18 runs. Both DRC mechanisms are purely defensive:
#    • the density-ratio TD weight down-weights exactly the novel syn coverage
#    • the 0.5 syn expectile leaks into real Q-targets via the shared V
#  This ablation removes both and isolates their effect.
#
#  CONFIG = the v3 DRC run, with EXACTLY two changes:
#    • --alpha 0.5  ->  0.15   (syn is a regularizer, can't dominate real)
#    • --sa_iql ... dropped    (-> falls back to standard IQL expectile 0.7,
#                                identical critic objective to the baseline)
#  Everything else (bc_weight, temperature, alpha warmup/ramp, critic arch,
#  reward_scale, PA loss) is held equal to the DRC run for a clean 3-way
#  comparison:  baseline  vs  v3 DRC  vs  Tier-1.
#
#  8 fresh runs (no --resume), 500k steps each, seed 0:
#    4 on medium-v2  +  4 on medium-replay-v2.
#  Writes to checkpoints/<env>/iql/augmented/alpha0.15/seed_0/ — a NEW dir,
#  no collision with the v3 alpha0.5 checkpoints.
#
#  Designed to run ALONGSIDE the v3 resume (scripts/launch_v3_resume.sh):
#    v3 resume = 3 procs/GPU (12 runs), this = 2 procs/GPU (8 runs) -> 5/GPU.
#  Run the v3 resume FIRST, then this, in the same salloc shell:
#      bash scripts/launch_v3_resume.sh
#      bash scripts/launch_tier1_ablation.sh
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
    echo "  If SSH disconnects, the salloc dies and every job is killed."
    echo "  Suggested: launch inside  'tmux new -s salloc'."
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

unset WANDB_MODE   # online; set WANDB_MODE=offline before running if init hangs

# Refuse to double-launch THIS ablation. We match on the wandb project string
# (unique to these runs) so we do NOT trip on the concurrent v3 procs.
n_running=$(pgrep -fc 'wandb_project disa-rl-tier1' || true)
if [ "${n_running:-0}" -gt 0 ]; then
    echo "ERROR: $n_running Tier-1 'train_iql.py' processes already running — refusing to launch."
    echo "Kill just these:  pkill -f 'wandb_project disa-rl-tier1'"
    exit 1
fi

# ── 1. Per-env config (must match the v3 DRC runs exactly) ─────────────────
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

# ── 2. Launch the 8 jobs (2 per GPU, alongside the v3 resume's 3/GPU) ──────
echo "=== Tier-1 ablation launching at $(date '+%F %T') ==="

# Tier-1 on medium-v2
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.15 --seed 0 \
        --num_steps 500000 --save_every 25000 --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-tier1 \
        > "logs/t1_${ENV}_s0.log" 2>&1 &
done

# Tier-1 on medium-replay-v2
for i in 0 1 2 3; do
    ENV=${ENVS_REPLAY[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.15 --seed 0 \
        --num_steps 500000 --save_every 25000 --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-tier1 \
        > "logs/t1_${ENV}_s0.log" 2>&1 &
done

sleep 30
n=$(pgrep -fc 'wandb_project disa-rl-tier1' || true)
echo
echo "=== launched ${n:-0} Tier-1 train_iql.py processes (target: 8) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

Tier-1 ablation started (running alongside the v3 resume).

  - Detach tmux with   Ctrl-b d   (KEEP the salloc alive)
  - Watch a log:       tail -F logs/t1_hopper-medium-v2_s0.log
  - Snapshot scores:   python scripts/aggregate_results.py --logs_glob "logs/t1_*.log"
  - Kill just Tier-1:  pkill -f 'wandb_project disa-rl-tier1'

EOM

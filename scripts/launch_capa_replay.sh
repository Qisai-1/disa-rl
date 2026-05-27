#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  DiSA-RL — CAPA sweep on the 4 medium-replay-v2 envs.
#
#  Method: CAPA (Critic-Anchored Proposal Augmentation, METHOD_V2_PROPOSAL.md §5)
#    - V & Q trained on REAL data only          (no syn-reward contamination)
#    - Actor AWR on MIXED batch, syn rows gated by exp(-β · Q-ensemble-std)
#    - BC anchor on real_batch (unchanged)
#    - Worst case: gate zeros syn rows → CAPA degenerates to real-only baseline.
#
#  Inputs: data/synthetic/<env>/synthetic_transitions.npz produced by
#  scripts/regen_syn_data_replay.sh (Phase A retrained v1 diffusion). Make
#  sure Phase C regen has finished before launching this:
#    [ $(ls data/synthetic/*-medium-replay-v2/synthetic_transitions.npz \
#        | wc -l) -eq 4 ]
#
#  Comparison set on medium-replay:
#    baseline (offline_only)  vs  v3 DRC (alpha 0.5 + SA-IQL)
#    vs  Tier-1 (alpha 0.15, no SA-IQL)  vs  CAPA (this).
#  Tier-1 already beat DRC on all 4 envs in the 2026-05-22 sweep
#  (hc 43.0, hopper 68.9, walker 83.0, ant 95.2). CAPA's structural
#  guarantee says it should match or beat Tier-1.
#
#  Config matches Tier-1 EXACTLY except for the two CAPA flags:
#    + --capa                  (real-only critic + syn-actor + uncertainty gate)
#    + --unc_beta 1.0          (geometric decay of syn weight with Q-std)
#  Output dir: checkpoints/<env>/iql/capa/alpha0.15/seed_0/  (separate from
#  Tier-1's .../augmented/alpha0.15/seed_0/ — no collision).
#
#  4 runs, 500k steps, seed 0, 1 GPU each.
#  Wall-clock estimate: ~13h per run (~21 it/s measured in smoke), all parallel.
#
#  USAGE:
#    bash scripts/launch_capa_replay.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

# ── 0. Environment ─────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

if ! timeout 60 python -c "import torch; assert torch.cuda.is_available(); torch.randn(8,device='cuda').sum().item()" 2>/dev/null; then
    echo "FATAL: CUDA not usable on $(hostname)."
    exit 1
fi
echo "preflight: CUDA OK — $(python -c 'import torch; print(torch.cuda.device_count())') GPU(s) visible"

# Refuse to launch if Phase C regen still running.
if pgrep -f 'generate_synthetic_data.py' > /dev/null; then
    echo "ERROR: generate_synthetic_data.py still running — Phase C regen not done."
    exit 1
fi

# Refuse to double-launch the CAPA sweep.
n_running=$(pgrep -fc 'wandb_project disa-rl-capa' || true)
if [ "${n_running:-0}" -gt 0 ]; then
    echo "ERROR: $n_running CAPA train_iql.py procs already running. Aborting."
    echo "Kill just CAPA:  pkill -f 'wandb_project disa-rl-capa'"
    exit 1
fi

ENVS=(
    halfcheetah-medium-replay-v2
    hopper-medium-replay-v2
    walker2d-medium-replay-v2
    ant-medium-replay-v2
)

# Sanity: confirm regen syn data exists.
for ENV in "${ENVS[@]}"; do
    SYN="data/synthetic/$ENV/synthetic_transitions.npz"
    if [ ! -f "$SYN" ]; then
        echo "FATAL: missing $SYN — run Phase C regen first."
        exit 1
    fi
done

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

unset WANDB_MODE
mkdir -p logs

echo
echo "=== CAPA sweep launching at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --capa --unc_beta 1.0 \
        --alpha 0.15 --seed 0 \
        --num_steps 500000 --save_every 25000 --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-capa \
        > "logs/capa_${ENV}_s0.log" 2>&1 &
    echo "  GPU $i  →  $ENV   (logs/capa_${ENV}_s0.log)"
done

# Brief settle so pgrep is reliable; replaces a longer sleep.
until [ "$(pgrep -fc 'wandb_project disa-rl-capa' || echo 0)" -ge 4 ]; do sleep 2; done

n=$(pgrep -fc 'wandb_project disa-rl-capa' || echo 0)
echo
echo "=== launched $n CAPA train_iql.py processes (target: 4) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

cat <<'EOM'

CAPA sweep started.

  - Watch a log:        tail -F logs/capa_hopper-medium-replay-v2_s0.log
  - Snapshot scores:    python scripts/aggregate_results.py --logs_glob "logs/capa_*.log"
  - Kill just CAPA:     pkill -f 'wandb_project disa-rl-capa'

EOM

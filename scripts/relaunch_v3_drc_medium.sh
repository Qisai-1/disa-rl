#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Relaunch ONLY the 4 v3 DRC medium-v2 runs.
#
#  Why: in the 2026-05-20 resume, these 4 runs picked a stale pre-v3
#  step_1000000.pt checkpoint (old narrow-Q architecture) and no-op'd out.
#  The stale checkpoints have since been moved to checkpoints/_stale_pre_v3/,
#  and train_iql.py --resume now selects by mtime — so they will correctly
#  resume from the v3 step_0100000.pt.
#
#  The other 8 jobs (4 baseline + 4 DRC-replay) are training fine — this
#  script does NOT touch them. Run it AFTER confirming the 4 broken DRC
#  jobs have exited (they self-exit immediately):
#      bash scripts/relaunch_v3_drc_medium.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
unset WANDB_MODE

ENVS_MEDIUM=( halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 )

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

# Guard: refuse if any of these 4 DRC medium-v2 runs is somehow still alive.
alive=$(pgrep -af 'iql/train_iql.py' | grep -E 'mode augmented' | grep -E 'medium-v2' | grep -vc 'replay' || true)
if [ "${alive:-0}" -gt 0 ]; then
    echo "ERROR: $alive DRC medium-v2 train_iql.py still running — refusing to double-launch."
    echo "Wait for them to exit, or inspect with: pgrep -af iql/train_iql.py"
    exit 1
fi

echo "=== relaunching 4 DRC medium-v2 runs at $(date '+%F %T') ==="
for i in 0 1 2 3; do
    ENV=${ENVS_MEDIUM[$i]}
    rs=$(reward_scale_for "$ENV")
    cs=$(critic_subset_for "$ENV")
    CUDA_VISIBLE_DEVICES=$i python -u iql/train_iql.py \
        --env "$ENV" --mode augmented --alpha 0.5 --seed 0 \
        --num_steps 500000 --resume --bc_weight 0.1 \
        --q_hidden_dims 256 256 256 256 --num_critics 10 --critic_subset "$cs" \
        --sa_iql --expectile_real 0.9 --expectile_syn 0.5 --sa_clip 0.5 2.0 \
        --action_noise_std 0.05 --temperature 3.0 \
        --alpha_warmup 50000 --alpha_ramp 50000 \
        --reward_scale "$rs" --pa_weight 0.001 --pa_min_q -100 \
        --wandb_project disa-rl-v3-drc \
        > "logs/v3_drc_${ENV}_s0.log" 2>&1 &
done

sleep 30
echo
echo "=== launched, checking resume lines (wandb.init may take ~3 min on a fresh node) ==="
sleep 120
grep -h "Resuming from\|Refusing to no-op" logs/v3_drc_*-medium-v2_s0.log 2>/dev/null || echo "(no resume line yet — re-check with: grep -h Resuming logs/v3_drc_*-medium-v2_s0.log)"
echo
echo "All 4 should say: Resuming from step_0100000.pt -> continuing at step 100,001/500,000"

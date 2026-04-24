#!/usr/bin/env bash
# =============================================================================
#  DiSA-RL Experiment Launcher
#  Usage: bash scripts/run_experiments.sh [MACHINE] [PHASE]
#
#  MACHINE: machine1 | machine2 | both (default: machine1)
#  PHASE:   diffusion | synthetic | iql | online | all (default: all)
#
#  Examples:
#    bash scripts/run_experiments.sh machine1 diffusion
#    bash scripts/run_experiments.sh machine2 iql
#    bash scripts/run_experiments.sh machine1 all
# =============================================================================

set -e
cd "$(dirname "$0")/.."   # always run from repo root

MACHINE=${1:-machine1}
PHASE=${2:-all}

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Environment assignment ────────────────────────────────────────────────────
# Machine 1 (titan3):       halfcheetah + hopper
# Machine 2 (scslab-titan1): walker2d + ant
if   [[ "$MACHINE" == "machine1" ]]; then
    ENVS=("halfcheetah-medium-v2" "hopper-medium-v2")
elif [[ "$MACHINE" == "machine2" ]]; then
    ENVS=("walker2d-medium-v2" "ant-medium-v2")
else
    error "Unknown machine: $MACHINE. Use machine1 or machine2."
fi

SEEDS=(0 1 2 3 4)
NUM_SEEDS=${#SEEDS[@]}

info "Machine  : $MACHINE"
info "Envs     : ${ENVS[*]}"
info "Phase    : $PHASE"
info "Seeds    : ${SEEDS[*]}"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────

wait_with_status() {
    local pids=("$@")
    local n=${#pids[@]}
    info "Waiting for $n background jobs..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            warn "Job $pid failed"
            ((failed++))
        fi
    done
    if [[ $failed -gt 0 ]]; then
        warn "$failed job(s) failed"
    else
        info "All $n jobs completed successfully"
    fi
}

check_ckpt() {
    local path=$1
    if [[ -f "$path" ]]; then
        info "Found: $path"
        return 0
    else
        warn "Missing: $path"
        return 1
    fi
}

# =============================================================================
#  PHASE 0: Diffusion training
# =============================================================================

phase_diffusion() {
    info "━━━ Phase 0: Diffusion Training ━━━"
    local pids=()

    for i in "${!ENVS[@]}"; do
        local env="${ENVS[$i]}"
        local ckpt="./checkpoints/$env/diffusion/offline_final.pt"

        if [[ -f "$ckpt" ]]; then
            info "Skipping $env — checkpoint exists"
            continue
        fi

        info "Starting diffusion training: $env"

        # Stagger launches by 2 min to avoid VRAM conflicts at startup
        if [[ $i -gt 0 ]]; then
            info "Waiting 120s before next launch..."
            sleep 120
        fi

        WANDB_MODE=offline python diffusion/train.py \
            --env        "$env" \
            --batch_size 256 \
            --lr         1e-4 \
            --num_steps  300000 \
            >> "logs/diffusion_${env}.log" 2>&1 &

        pids+=($!)
        info "$env → PID $! | log: logs/diffusion_${env}.log"
    done

    [[ ${#pids[@]} -gt 0 ]] && wait_with_status "${pids[@]}"
    info "━━━ Phase 0 complete ━━━"
}

# =============================================================================
#  PHASE 1: Synthetic data generation
# =============================================================================

phase_synthetic() {
    info "━━━ Phase 1: Synthetic Data Generation ━━━"

    for env in "${ENVS[@]}"; do
        local out="./data/synthetic/$env/synthetic_transitions.npz"
        local ckpt="./checkpoints/$env/diffusion/offline_final.pt"

        if [[ -f "$out" ]]; then
            info "Skipping $env — synthetic data exists"
            continue
        fi

        if [[ ! -f "$ckpt" ]]; then
            error "Diffusion checkpoint missing: $ckpt — run phase_diffusion first"
        fi

        info "Generating synthetic data: $env"
        python generate_synthetic_data.py \
            --env          "$env" \
            --n_transitions 1000000 \
            --batch_size   64 \
            >> "logs/synthetic_${env}.log" 2>&1

        # Verify output
        if [[ -f "$out" ]]; then
            local size
            size=$(du -sh "$out" | cut -f1)
            info "$env → $out ($size)"
        else
            error "Generation failed for $env"
        fi

        # Validate rewards
        info "Validating rewards: $env"
        python reward_computer.py --env "$env" --test_analytic \
            >> "logs/synthetic_${env}.log" 2>&1
    done

    info "━━━ Phase 1 complete ━━━"
}

# =============================================================================
#  PHASE 2: Offline IQL training (augmented)
# =============================================================================

phase_iql() {
    info "━━━ Phase 2: Offline IQL Training ━━━"
    local pids=()

    for env in "${ENVS[@]}"; do
        # Use augmented if synthetic data exists, otherwise offline_only
        local syn="./data/synthetic/$env/synthetic_transitions.npz"
        local mode="offline_only"
        if [[ -f "$syn" ]]; then
            mode="augmented"
            info "IQL augmented: $env (synthetic data found)"
        else
            warn "IQL offline_only: $env (no synthetic data — run phase_synthetic first)"
        fi

        for seed in "${SEEDS[@]}"; do
            WANDB_MODE=offline python iql/train_iql.py \
                --env       "$env" \
                --mode      "$mode" \
                --seed      "$seed" \
                --num_steps 1000000 \
                >> "logs/iql_${env}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done

    info "Launched ${#pids[@]} IQL jobs"
    wait_with_status "${pids[@]}"
    info "━━━ Phase 2 complete ━━━"
}

# =============================================================================
#  PHASE 3: Online RL training
# =============================================================================

phase_online() {
    info "━━━ Phase 3: Online RL Training ━━━"
    local pids=()

    for env in "${ENVS[@]}"; do
        local iql_ckpt="./checkpoints/$env/iql/augmented/best.pt"
        local diff_ckpt="./checkpoints/$env/diffusion/offline_final.pt"
        local syn="./data/synthetic/$env/synthetic_transitions.npz"

        check_ckpt "$iql_ckpt"  || error "Run phase_iql first"
        check_ckpt "$syn"       || error "Run phase_synthetic first"

        for seed in "${SEEDS[@]}"; do
            info "Online SAC: $env  seed=$seed"

            WANDB_MODE=offline python online_rl/train_online.py \
                --env              "$env" \
                --iql_ckpt         "$iql_ckpt" \
                --diffusion_ckpt   "$diff_ckpt" \
                --synthetic_data   "$syn" \
                --num_steps        500000 \
                --seed             "$seed" \
                >> "logs/online_${env}_s${seed}.log" 2>&1 &

            pids+=($!)
        done
    done

    info "Launched ${#pids[@]} online jobs"
    wait_with_status "${pids[@]}"
    info "━━━ Phase 3 complete ━━━"
}

# =============================================================================
#  PHASE 4: Ablations (machine 1 only, halfcheetah)
# =============================================================================

phase_ablations() {
    [[ "$MACHINE" != "machine1" ]] && { info "Ablations run on machine1 only — skipping"; return; }
    info "━━━ Phase 4: Ablations ━━━"

    local env="halfcheetah-medium-v2"
    local syn="./data/synthetic/$env/synthetic_transitions.npz"
    local pids=()

    # Ablation 1: fixed alpha values
    for alpha in 0.3 0.5 0.7; do
        for seed in 0 1 2; do
            info "Ablation fixed_alpha=$alpha  seed=$seed"
            WANDB_MODE=offline python iql/train_iql.py \
                --env   "$env" \
                --mode  augmented \
                --alpha "$alpha" \
                --seed  "$seed" \
                >> "logs/ablation_alpha${alpha}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done

    # Ablation 2: no augmentation baseline
    for seed in 0 1 2; do
        info "Ablation offline_only  seed=$seed"
        WANDB_MODE=offline python iql/train_iql.py \
            --env  "$env" \
            --mode offline_only \
            --seed "$seed" \
            >> "logs/ablation_offline_only_s${seed}.log" 2>&1 &
        pids+=($!)
    done

    info "Launched ${#pids[@]} ablation jobs"
    wait_with_status "${pids[@]}"
    info "━━━ Phase 4 complete ━━━"
}

# =============================================================================
#  STATUS CHECK
# =============================================================================

phase_status() {
    info "━━━ Experiment Status ━━━"
    echo ""

    for env in "${ENVS[@]}"; do
        echo "  [$env]"

        # Diffusion
        local d="./checkpoints/$env/diffusion/offline_final.pt"
        [[ -f "$d" ]] && echo "    ✓ Diffusion: done" || echo "    ✗ Diffusion: missing"

        # Synthetic data
        local s="./data/synthetic/$env/synthetic_transitions.npz"
        if [[ -f "$s" ]]; then
            local sz; sz=$(du -sh "$s" | cut -f1)
            echo "    ✓ Synthetic: done ($sz)"
        else
            echo "    ✗ Synthetic: missing"
        fi

        # IQL seeds
        local iql_done=0
        for seed in "${SEEDS[@]}"; do
            local f="./checkpoints/$env/iql/augmented/seed_${seed}"
            [[ -f "$f/final.pt" ]] && ((iql_done++))
        done
        echo "    IQL augmented: $iql_done/$NUM_SEEDS seeds done"

        # Online seeds
        local online_done=0
        for seed in "${SEEDS[@]}"; do
            local f="./checkpoints/$env/online/seed_${seed}/final.pt"
            [[ -f "$f" ]] && ((online_done++))
        done
        echo "    Online SAC:    $online_done/$NUM_SEEDS seeds done"
        echo ""
    done
}

# =============================================================================
#  MAIN
# =============================================================================

# Create log directory
mkdir -p logs

case "$PHASE" in
    diffusion)  phase_diffusion ;;
    synthetic)  phase_synthetic ;;
    iql)        phase_iql ;;
    online)     phase_online ;;
    ablations)  phase_ablations ;;
    status)     phase_status ;;
    all)
        phase_status
        phase_diffusion
        phase_synthetic
        phase_iql
        phase_online
        phase_ablations
        phase_status
        info "All phases complete."
        ;;
    *)
        error "Unknown phase: $PHASE. Use: diffusion | synthetic | iql | online | ablations | status | all"
        ;;
esac
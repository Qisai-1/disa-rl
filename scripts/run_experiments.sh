#!/usr/bin/env bash
# Usage: bash scripts/run_experiments.sh [machine1|machine2] [diffusion|synthetic|iql|online|ablations|status|all]
set -e; cd "$(dirname "$0")/.."
MACHINE=${1:-machine1}; PHASE=${2:-all}
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}[INFO]${N}  $*"; }
warn()  { echo -e "${Y}[WARN]${N}  $*"; }
error() { echo -e "${R}[ERROR]${N} $*"; exit 1; }
[[ "$MACHINE" == "machine1" ]] && ENVS=("halfcheetah-medium-v2" "hopper-medium-v2") \
|| [[ "$MACHINE" == "machine2" ]] && ENVS=("walker2d-medium-v2" "ant-medium-v2") \
|| error "Unknown machine: $MACHINE"
SEEDS=(0 1 2 3 4); mkdir -p logs
wait_jobs() { local f=0; for p in "$@"; do wait "$p" || ((f++)); done; [[ $f -gt 0 ]] && warn "$f failed" || info "All done."; }

phase_status() {
    info "━━━ Status: $MACHINE ━━━"
    for env in "${ENVS[@]}"; do
        echo "  [$env]"
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] && echo "    ✓ Diffusion" || echo "    ✗ Diffusion"
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] && echo "    ✓ Synthetic" || echo "    ✗ Synthetic"
        local i=0 o=0
        for s in "${SEEDS[@]}"; do
            [[ -f "checkpoints/$env/iql/augmented/seed_${s}/final.pt" ]] && ((i++))
            [[ -f "checkpoints/$env/online/seed_${s}/final.pt" ]] && ((o++))
        done
        echo "    IQL augmented: $i/${#SEEDS[@]}  |  Online: $o/${#SEEDS[@]}"
    done
}

phase_diffusion() {
    info "━━━ Phase 0: Diffusion Training ━━━"; local pids=()
    for i in "${!ENVS[@]}"; do
        local env="${ENVS[$i]}"
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] && { info "Skip $env"; continue; }
        [[ $i -gt 0 ]] && sleep 120
        WANDB_MODE=offline python diffusion/train.py --env "$env" --batch_size 256 --lr 1e-4 --num_steps 300000 >> "logs/diffusion_${env}.log" 2>&1 &
        pids+=($!); info "$env → PID $! | logs/diffusion_${env}.log"
    done
    [[ ${#pids[@]} -gt 0 ]] && wait_jobs "${pids[@]}"
}

phase_synthetic() {
    info "━━━ Phase 1: Synthetic Data ━━━"
    for env in "${ENVS[@]}"; do
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] && { info "Skip $env"; continue; }
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] || error "No diffusion ckpt for $env"
        info "Generating: $env"
        python generate_synthetic_data.py --env "$env" --n_transitions 1000000 >> "logs/synthetic_${env}.log" 2>&1
        python reward_computer.py --env "$env" --test_analytic >> "logs/synthetic_${env}.log" 2>&1
        info "$env done"
    done
}

phase_iql() {
    info "━━━ Phase 2: Offline IQL ━━━"; local pids=()
    for env in "${ENVS[@]}"; do
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] || error "No synthetic data for $env"
        for seed in "${SEEDS[@]}"; do
            WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode augmented --seed "$seed" --num_steps 1000000 >> "logs/iql_${env}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done
    info "Launched ${#pids[@]} IQL jobs"; wait_jobs "${pids[@]}"
}

phase_online() {
    info "━━━ Phase 3: Online RL ━━━"; local pids=()
    for env in "${ENVS[@]}"; do
        local iq="checkpoints/$env/iql/augmented/best.pt"
        local df="checkpoints/$env/diffusion/offline_final.pt"
        local sy="data/synthetic/$env/synthetic_transitions.npz"
        [[ -f "$iq" ]] || error "No IQL ckpt: $iq"; [[ -f "$sy" ]] || error "No synthetic: $sy"
        for seed in "${SEEDS[@]}"; do
            WANDB_MODE=offline python online_rl/train_online.py --env "$env" --iql_ckpt "$iq" --diffusion_ckpt "$df" --synthetic_data "$sy" --num_steps 500000 --seed "$seed" >> "logs/online_${env}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done
    info "Launched ${#pids[@]} online jobs"; wait_jobs "${pids[@]}"
}

phase_ablations() {
    [[ "$MACHINE" == "machine1" ]] || { info "Ablations: machine1 only"; return; }
    info "━━━ Phase 4: Ablations ━━━"; local pids=()
    local env="halfcheetah-medium-v2"
    for alpha in 0.3 0.5 0.7; do for seed in 0 1 2; do
        WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode augmented --alpha "$alpha" --seed "$seed" >> "logs/ablation_alpha${alpha}_s${seed}.log" 2>&1 & pids+=($!)
    done; done
    for seed in 0 1 2; do
        WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode offline_only --seed "$seed" >> "logs/ablation_offline_s${seed}.log" 2>&1 & pids+=($!)
    done
    wait_jobs "${pids[@]}"
}

case "$PHASE" in
    diffusion) phase_diffusion ;;  synthetic) phase_synthetic ;;
    iql)       phase_iql ;;        online)    phase_online ;;
    ablations) phase_ablations ;;  status)    phase_status ;;
    all) phase_status; phase_diffusion; phase_synthetic; phase_iql; phase_online; phase_ablations; phase_status ;;
    *) error "Unknown phase: $PHASE" ;;
esac

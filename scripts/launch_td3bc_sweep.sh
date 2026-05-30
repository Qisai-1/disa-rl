#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TD3+BC sweep — second backbone for the AAAI table.
#
#  GTA (NeurIPS'24) reports backbone-agnostic gains on TD3+BC. TD3+BC is
#  often the strongest published baseline on D4RL medium-replay
#  (hopper ~66, walker ~84). This sweep gives us a TD3+BC column for the
#  AAAI table — both offline_only and DiSA-augmented (real-only critic +
#  mixed actor + uncertainty gate; same hooks as CAPA but ported to the
#  deterministic TD3+BC actor).
#
#  Composes with the new --reward_norm corl + --obs_norm flags. TD3+BC's
#  λ = α / E[|Q|] auto-normalizes Q scale so the BC term is comparable —
#  one less hand-tuned hyperparameter than IQL.
#
#  USAGE:
#    bash scripts/launch_td3bc_sweep.sh
#
#  Output:
#    logs/td3bc_<env>_<method>_s<SEED>.log
#    wandb project: disa-rl-td3bc
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: iql/train_iql.py running (GPUs busy)."
    exit 1
fi
for e in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    [ -f "data/synthetic_gta/$e/synthetic_transitions.npz" ] || {
        echo "FATAL: missing GTA syn data for $e"; exit 1
    }
done

mkdir -p logs
COMMON="--backbone td3bc --num_steps 500000 --save_every 100000 \
        --eval_every 10000 --reward_norm corl --obs_norm \
        --num_critics 10 --critic_subset 2 \
        --td3bc_alpha 2.5 --td3bc_policy_freq 2 \
        --q_hidden_dims 256 256 --wandb_project disa-rl-td3bc"
AUG="--mode augmented --alpha 0.5 --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"

i=0
run () {  # env method seed
  local env=$1 method=$2 seed=$3 gpu=$(( i % 4 )); i=$((i+1))
  local tag="${env%%-*}_${method}_s${seed}"
  case $method in
    offline)   flags="--mode offline_only --bc_weight 0.1" ;;
    aug)       flags="$AUG --td3bc_unc_beta 0 \
                     --synthetic_data data/synthetic_gta/${env}/synthetic_transitions.npz" ;;
    augcapa)   flags="$AUG --td3bc_unc_beta 1.0 \
                     --synthetic_data data/synthetic_gta/${env}/synthetic_transitions.npz" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $flags $COMMON > "logs/td3bc_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  $method  seed=$seed"
}

echo "=== TD3+BC sweep launching $(date '+%F %T') ==="
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for method in offline aug augcapa; do
    for seed in 0 1 2; do
      run "$env" "$method" "$seed"
    done
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-td3bc')" -ge 18 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-td3bc')/18 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

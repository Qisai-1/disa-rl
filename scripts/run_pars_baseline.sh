#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Reproduce PARS (ICML 2025) numbers on D4RL MuJoCo medium-v2.
#  Specifically validate their headline claim:  hopper-medium-v2 = 104.1
#
#  PARS uses JAX, so this sets up a separate conda env to avoid breaking
#  our PyTorch env. Takes ~1-2 hours to set up, ~6-12h per env to train.
#
#  Usage:
#      bash scripts/run_pars_baseline.sh             # all 4 envs
#      bash scripts/run_pars_baseline.sh hopper      # one env only
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd /work/mech-ai-scratch/supersai

PARS_DIR="$PWD/pars_baseline"
PARS_ENV="pars-jax"

# ── 1. Clone PARS if not already there ─────────────────────────────────────
if [ ! -d "$PARS_DIR" ]; then
    git clone https://github.com/LGAI-Research/pars "$PARS_DIR"
fi
cd "$PARS_DIR"

# ── 2. Create JAX env (separate from our 'disa' env) ───────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^$PARS_ENV "; then
    echo "Creating fresh conda env: $PARS_ENV"
    conda create -y -n "$PARS_ENV" python=3.10
    conda activate "$PARS_ENV"
    pip install --upgrade pip
    # PARS uses JAX; install GPU build for CUDA 12.x to match A100 driver
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
        pip install "jax[cuda12]"
    # Their other deps
    pip install -r requirements.txt 2>/dev/null || \
        pip install flax optax tqdm wandb hydra-core gym==0.23 mujoco==2.3.6
    # D4RL
    pip install git+https://github.com/Farama-Foundation/d4rl@master
else
    conda activate "$PARS_ENV"
fi

# ── 3. Pick env(s) ─────────────────────────────────────────────────────────
ENVS=("$@")
if [ ${#ENVS[@]} -eq 0 ]; then
    ENVS=(hopper walker2d halfcheetah ant)
fi

# ── 4. Run PARS on each env ────────────────────────────────────────────────
# Their configs target medium-expert by default; we override env to medium-v2.
mkdir -p logs
for env_short in "${ENVS[@]}"; do
    case "$env_short" in
        hopper)      env_full="hopper-medium-v2"      ; rs=10 ;;
        walker2d)    env_full="walker2d-medium-v2"    ; rs=10 ;;
        halfcheetah) env_full="halfcheetah-medium-v2" ; rs=5  ;;
        ant)         env_full="ant-medium-v2"         ; rs=5  ;;
        *) echo "skipping unknown env: $env_short"; continue ;;
    esac
    log="logs/pars_${env_full}.log"
    echo "=== PARS  $env_full  reward_scale=$rs  → $log ==="
    # PARS main.py args may differ slightly per version; this is a best-effort
    # template. Check ./main.py --help if it fails.
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
        env="$env_full" \
        reward_scale=$rs \
        seed=0 \
        wandb_project=disa-rl-pars-baseline \
        > "$log" 2>&1 &
done

wait
echo
echo "=== All PARS runs done at $(date) ==="
echo "Final scores (last 'normalized_return' line per log):"
for log in logs/pars_*.log; do
    score=$(grep -i "normalized" "$log" | tail -1)
    echo "  $(basename $log): $score"
done

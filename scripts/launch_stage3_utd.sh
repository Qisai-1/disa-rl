#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3: UTD>1 (RLPD/EDAC-style) on the BEST stage-2 config.
#
#  Composes:
#    - CAPA+GTA + critic_syn_coef (capa_plus gates syn into V/Q critic)
#    - alpha (real/syn actor sampling ratio) and coef chosen from stage-2
#    - --utd N : k-1 critic-only updates per actor update (default 4 = RLPD)
#    - LayerNorm critic + 10-critic ensemble + reward_scale 1 (stable config)
#    - GTA-style amplified-return syn data with velocity-integration physics fix
#
#  USAGE:
#    bash scripts/launch_stage3_utd.sh <ALPHA> <COEF> [UTD]
#    e.g.  bash scripts/launch_stage3_utd.sh 0.5 0.25 4
#
#  Output:
#    logs/s3_<env>_a<ALPHA>_c<COEF>_u<UTD>_s<SEED>.log
#    wandb project: disa-rl-s3
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa

ALPHA=${1:-0.5}
COEF=${2:-0.25}
UTD=${3:-4}

# Refuse if any iql/train_iql.py is currently running (GPUs busy).
if pgrep -f 'python -u iql/train_iql.py' >/dev/null; then
    echo "ERROR: a python iql/train_iql.py is already running (GPUs busy)."
    echo "       Stage 3 needs the GPUs. Wait for stage 2 to finish."
    exit 1
fi

for e in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    [ -f "data/synthetic_gta/$e/synthetic_transitions.npz" ] || {
        echo "FATAL: missing GTA syn data for $e (data/synthetic_gta/$e/synthetic_transitions.npz)"
        exit 1
    }
done

mkdir -p logs
COMMON="--num_steps 500000 --save_every 100000 --eval_every 10000 \
        --expectile 0.7 --temperature 3.0 --reward_scale 1 --q_hidden_dims 256 256 \
        --wandb_project disa-rl-s3"
# NOTE: --save_every 100000 (vs Stage 2's 999999999) so the Stage 3 winner
#       produces a fresh CAPA+ checkpoint that can seed the O2O pivot.
AUG="--mode augmented --alpha ${ALPHA} --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef ${COEF} \
      --num_critics 10 --critic_subset 2 --utd ${UTD}"

i=0
run () {  # env seed
  local env=$1 seed=$2 gpu=$(( i % 4 )); i=$((i+1))
  local tag="${env%%-*}_a${ALPHA}_c${COEF}_u${UTD}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
      $AUG $CAPA \
      --synthetic_data "data/synthetic_gta/${env}/synthetic_transitions.npz" \
      $COMMON > "logs/s3_${tag}.log" 2>&1 &
  echo "  GPU$gpu  $env  seed=$seed  alpha=$ALPHA coef=$COEF utd=$UTD"
}

echo "=== Stage 3 (UTD=${UTD}) launching $(date '+%F %T') ==="
echo "    alpha=${ALPHA}  critic_syn_coef=${COEF}"
for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
  for seed in 0 1 2; do
    run "$env" "$seed"
  done
done

until [ "$(pgrep -fc 'wandb_project disa-rl-s3')" -ge 6 ]; do sleep 3; done
echo "launched $(pgrep -fc 'wandb_project disa-rl-s3')/6 runs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "aggregate with:"
echo "  python - <<'PY'"
echo "  import glob,re,numpy as np;from collections import defaultdict"
echo "  rn=re.compile(r'normalized=\s*([-\d.]+)');rs=re.compile(r'(\d+)/[0-9]+')"
echo "  c,s=defaultdict(list),defaultdict(list)"
echo "  for f in sorted(glob.glob('logs/s3_*.log')):"
echo "      n=f.split('/')[-1].replace('.log',''); txt=open(f,errors='replace').read()"
echo "      v=[float(x) for x in rn.findall(txt)]; st=rs.findall(txt)"
echo "      if v: c[n].append(np.mean(v[-10:])); s[n].append(int(st[-1]) if st else 0)"
echo "  for k in sorted(c): print(f'  {k:<48} last10avg={np.mean(c[k]):.1f}  step={max(s[k]):>7,}')"
echo "  PY"

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Auto-chain #6: post-Stage-3.0 → multi-seed extension OR fallback debug.
#
#  If Stage 3.0 (ensemble-filtered) shows a clear lift over Stage 2.7
#  (raw GTA), run a 5-seed multi-seed extension at the winning config to
#  pin down significance for the paper.
#
#  If no lift, write a notes doc and stop — user reviews to decide.
#
#  USAGE:
#    NGPU=3 nohup bash scripts/auto_chain_post_s3p0.sh > logs/auto_chain_6.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disa
NGPU=${NGPU:-3}

echo "[$(date '+%F %T')] Auto-chain-6 armed; waiting for Stage 3.0 to finish..."

while [ "$(pgrep -fc 'wandb_project disa-rl-s3p0')" -eq 0 ]; do sleep 60; done
echo "[$(date '+%F %T')] Stage 3.0 detected; waiting for completion..."
while [ "$(pgrep -fc 'wandb_project disa-rl-s3p0')" -gt 0 ]; do sleep 60; done
echo "[$(date '+%F %T')] Stage 3.0 done. Aggregating..."

python scripts/report_runs.py 'logs/s3p0_*.log' > results/stage3p0_final.txt 2>&1 || true
python scripts/report_runs.py 'logs/s2p7_*.log' > results/stage2p7_final.txt 2>&1 || true
cat results/stage3p0_final.txt | tail -15

# Compute the lift of Stage 3.0 over Stage 2.7
python - <<'PY' | tee results/s3p0_vs_s2p7_verdict.txt
import re, glob, os
import numpy as np
from collections import defaultdict
def extract(path):
    rn = re.compile(r'normalized=\s*([-\d.]+)')
    txt = open(path, errors='replace').read().replace('\r', '\n')
    vals=[]
    for line in txt.split('\n'):
        m=rn.search(line)
        if m: vals.append(float(m.group(1)))
    return vals
def by_env(pat, k=10):
    out=defaultdict(list)
    for f in sorted(glob.glob(pat)):
        n=os.path.basename(f).replace('.log','')
        env=n.split('_')[1]
        v=extract(f)
        if len(v)>=k: out[env].append(np.mean(v[-k:]))
    return out
s27 = by_env('logs/s2p7_*.log')
s30 = by_env('logs/s3p0_*.log')
print("env       s27 (raw GTA)   s30 (ens-filtered)   Δ")
wins=0
for env in sorted(set(s27)|set(s30)):
    a=np.mean(s27.get(env,[float('nan')])); b=np.mean(s30.get(env,[float('nan')]))
    d=b-a
    tag = "WIN" if d>=3 else ("parity" if d>=-1 else "regression")
    print(f"  {env}: {a:.2f}  →  {b:.2f}   Δ={d:+.2f}  {tag}")
    if d>=3: wins+=1
print(f"\nVERDICT_S30_VS_S27={'WIN' if wins>=2 else ('MIXED' if wins>=1 else 'NO_LIFT')}")
PY

VERDICT=$(grep -oE 'VERDICT_S30_VS_S27=[A-Z_]+' results/s3p0_vs_s2p7_verdict.txt | cut -d= -f2)
echo "[$(date '+%F %T')] Diffusion ensemble verdict: $VERDICT"

# Sanity
if pgrep -f 'python -u (iql|diffusion)/train.py' >/dev/null; then
    echo "[$(date '+%F %T')] procs still alive — abort"; exit 1
fi

case $VERDICT in
  WIN|MIXED)
    echo "[$(date '+%F %T')] Lift detected. Launching multi-seed extension (5 seeds × 2 envs)..."
    # 10 runs of the winning combo (Stage 3.0 config) with seeds 0..4
    # (seeds 0,1,2 are already done from S3.0; only run 3,4 for 4 new runs)
    COMMON="--num_steps 500000 --save_every 100000 --eval_every 10000 \
            --expectile 0.7 --temperature 3.0 --reward_scale 1 \
            --reward_norm corl --obs_norm --no-syn_normalize_rewards \
            --utd 4 \
            --q_hidden_dims 256 256 --wandb_project disa-rl-multiseed"
    AUG="--mode augmented --alpha 0.5 --bc_weight 0.1 --alpha_warmup 50000 --alpha_ramp 50000"
    CAPA="--capa --capa_plus --unc_beta 1.0 --critic_syn_coef 0.25 \
          --num_critics 10 --critic_subset 2"
    NOVEL="--awr_gate_mode temper --gbc_weight 0.05 --gbc_gate_min 0.6 \
           --asym_expectile_syn --critic_syn_coef_warmup 50000"
    i=0
    for env in hopper-medium-replay-v2 walker2d-medium-replay-v2; do
        for seed in 3 4; do
            gpu=$(( i % NGPU )); i=$((i+1))
            CUDA_VISIBLE_DEVICES=$gpu python -u iql/train_iql.py --env "$env" --seed "$seed" \
                $AUG $CAPA $NOVEL $COMMON \
                --synthetic_data "data/synthetic_ens_filtered/${env}/synthetic_transitions.npz" \
                > "logs/ms_${env%%-*}_s${seed}.log" 2>&1 &
            echo "  GPU$gpu  $env  seed=$seed"
        done
    done
    until [ "$(pgrep -fc 'wandb_project disa-rl-multiseed')" -ge 4 ]; do sleep 3; done
    echo "[$(date '+%F %T')] Multi-seed extension launched (4 new runs; total 5 seeds with S3.0's 0-2)"
    ;;
  NO_LIFT)
    echo "[$(date '+%F %T')] No lift from ensemble filter. Writing debug note."
    cat > results/s3p0_no_lift.txt <<EOF
Stage 3.0 (ensemble-filtered syn) did not lift over Stage 2.7 (raw GTA syn).
Possible reasons:
  - keep_quantile=0.5 too aggressive or too lax
  - Ensemble disagreement metric (per-row obs variance) not informative
  - Already at the data-processing ceiling for these envs
Recommended next step: examine data/synthetic_ens_filtered/<env>/disagreement_stats.npz
to see disagreement distribution. Try keep_quantile=0.25 (stricter) or 0.75 (lax).
EOF
    cat results/s3p0_no_lift.txt
    ;;
esac
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "[$(date '+%F %T')] === Auto-chain-6 complete ==="

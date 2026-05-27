#!/bin/bash
# Quick snapshot of the current run state. Run from project root.
# Usage:  bash scripts/status.sh
cd "$(dirname "$0")/.."

echo "=== $(date '+%F %T')  ==="
echo

echo "── GPUs ──────────────────────────────────────────────"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
           --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable from this node)"
echo

echo "── Python processes (this user) ──────────────────────"
ps -u "$USER" -o pid,etime,pcpu,pmem,cmd --sort=-pcpu 2>/dev/null \
  | awk 'NR==1 || /python|train_iql|eval_final|generate_synthetic|diffusion\/train/' \
  | head -20
echo

echo "── Eval/Gen/Sanity logs (newest 12) ──────────────────"
ls -lt logs/ 2>/dev/null | head -13 | tail -12
echo

echo "── Latest tail of each active log ────────────────────"
for f in $(ls -t logs/eval_*-medium*.log logs/gen_*.log logs/sanity_*.log \
              logs/vcdg_*.log logs/offline_*.log 2>/dev/null | head -8); do
  echo "── $f"
  # Look for last meaningful line
  tail -3 "$f" 2>/dev/null
  echo
done

echo "── Result CSVs ───────────────────────────────────────"
ls -la results/*.csv 2>/dev/null | awk '{print $5, $9, $6, $7, $8}'
echo

echo "── New eval scores in last 10 minutes ────────────────"
find logs -name '*.log' -mmin -10 2>/dev/null \
  | xargs -I{} grep -H "normalized=" {} 2>/dev/null \
  | tail -10

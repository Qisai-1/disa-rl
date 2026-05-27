#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  DiSA-RL — Local IQL Training Script
#  Works on any single-GPU machine (titan3, scslab-titan1, beast0)
#
#  Usage:
#    bash scripts/run_iql_local.sh
#
#  Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these values
# ══════════════════════════════════════════════════════════════════════════════

ENVS="halfcheetah-medium-v2 hopper-medium-v2"
ALPHAS="0.5 0.25 0.0"
SEEDS="0 1 2 3 4"
MODE="augmented"
BC_WEIGHT=0.1
WANDB_PROJECT="disa-rl-medium"
MAX_PARALLEL=10
NUM_STEPS=1000000
GPU=0

# ══════════════════════════════════════════════════════════════════════════════
#  SCRIPT — no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
mkdir -p logs results

# Timestamped results file
RESULTS_FILE="results/iql_results_$(date +%Y%m%d_%H%M%S).txt"
RESULTS_CSV="results/iql_results_$(date +%Y%m%d_%H%M%S).csv"

log_and_print() {
    echo "$1"
    echo "$1" >> "$RESULTS_FILE"
}

echo "=========================================="  | tee "$RESULTS_FILE"
echo "  DiSA-RL Local IQL Training"               | tee -a "$RESULTS_FILE"
echo "  $(date)"                                   | tee -a "$RESULTS_FILE"
echo "=========================================="  | tee -a "$RESULTS_FILE"
echo "  Envs    : $ENVS"                           | tee -a "$RESULTS_FILE"
echo "  Alphas  : $ALPHAS"                         | tee -a "$RESULTS_FILE"
echo "  Seeds   : $SEEDS"                          | tee -a "$RESULTS_FILE"
echo "  Mode    : $MODE"                           | tee -a "$RESULTS_FILE"
echo "  BC wt   : $BC_WEIGHT"                      | tee -a "$RESULTS_FILE"
echo "  GPU     : $GPU"                            | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"
echo ""

# Verify synthetic data
if [[ "$MODE" == "augmented" ]]; then
    echo "Checking synthetic data..."
    for env in $ENVS; do
        syn="./data/synthetic/$env/synthetic_transitions.npz"
        if [[ -f "$syn" ]]; then
            echo "  ✓ $env"
        else
            echo "  ✗ $env — MISSING: $syn"
            echo "    Run: python generate_synthetic_data.py --env $env"
            exit 1
        fi
    done
    echo ""
fi

# Training loop
for alpha in $ALPHAS; do
    echo "------------------------------------------" | tee -a "$RESULTS_FILE"
    echo "  Alpha = $alpha  ($(date))"               | tee -a "$RESULTS_FILE"
    echo "------------------------------------------" | tee -a "$RESULTS_FILE"

    pids=()
    for env in $ENVS; do
        for seed in $SEEDS; do
            # Wait if we have hit the parallel limit
            while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
                new_pids=()
                for pid in "${pids[@]}"; do
                    kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
                done
                pids=("${new_pids[@]}")
                [[ ${#pids[@]} -ge $MAX_PARALLEL ]] && sleep 10
            done

            log="logs/iql_${env}_${MODE}_alpha${alpha}_s${seed}.log"
            echo "  Launching: $env alpha=$alpha seed=$seed → $log"

            CUDA_VISIBLE_DEVICES=$GPU python iql/train_iql.py \
                --env            "$env" \
                --mode           "$MODE" \
                --alpha          "$alpha" \
                --bc_weight      "$BC_WEIGHT" \
                --seed           "$seed" \
                --num_steps      "$NUM_STEPS" \
                --expectile      "$EXPECTILE" \
                --temperature    "$TEMPERATURE" \
                --wandb_project  "$WANDB_PROJECT" \
                >> "$log" 2>&1 &
            pids+=($!)
        done
    done

    # Wait for all jobs at this alpha before moving to next
    echo "  Waiting for ${#pids[@]} jobs to finish..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "  Done alpha=$alpha  ($(date))" | tee -a "$RESULTS_FILE"
    echo ""
done

echo "==========================================" | tee -a "$RESULTS_FILE"
echo "  All training done: $(date)"               | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"

# ── Results summary ────────────────────────────────────────────────────────
{
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY"
echo "=========================================="
printf "  %-35s %7s %10s %10s\n" "Env" "Alpha" "Best" "Final"
printf "  %s\n" "------------------------------------------------------------"
} | tee -a "$RESULTS_FILE"

# CSV header
echo "env,alpha,seeds,best_mean,best_std,final_mean,final_std" > "$RESULTS_CSV"

for env in $ENVS; do
    for alpha in $ALPHAS; do
        best_scores=()
        final_scores=()

        for seed in $SEEDS; do
            log="logs/iql_${env}_${MODE}_alpha${alpha}_s${seed}.log"
            [[ ! -f "$log" ]] && continue

            best=$(grep -oP "normalized=\K[0-9.]+" "$log" 2>/dev/null | \
                   sort -n | tail -1)
            final=$(grep -oP "normalized=\K[0-9.]+" "$log" 2>/dev/null | \
                    tail -1)

            [[ -n "$best"  ]] && best_scores+=("$best")
            [[ -n "$final" ]] && final_scores+=("$final")
        done

        if [[ ${#best_scores[@]} -gt 0 ]]; then
            result=$(python3 -c "
import numpy as np
best  = [${best_scores[@]}]
final = [${final_scores[@]}]
bm, bs = np.mean(best), np.std(best)
fm, fs = np.mean(final), np.std(final)
print(f'DISPLAY {bm:.1f}±{bs:.1f} {fm:.1f}±{fs:.1f}')
print(f'CSV {len(best)},{bm:.3f},{bs:.3f},{fm:.3f},{fs:.3f}')
" 2>/dev/null)
            display=$(echo "$result" | grep "^DISPLAY" | sed 's/DISPLAY //')
            csv_vals=$(echo "$result" | grep "^CSV" | sed 's/CSV //')
            best_mean=$(echo "$display" | awk '{print $1}')
            final_mean=$(echo "$display" | awk '{print $2}')

            printf "  %-35s %7s %10s %10s\n" "$env" "$alpha" "$best_mean" "$final_mean" \
                | tee -a "$RESULTS_FILE"
            echo "$env,$alpha,$csv_vals" >> "$RESULTS_CSV"
        else
            printf "  %-35s %7s %10s %10s\n" "$env" "$alpha" "---" "---" \
                | tee -a "$RESULTS_FILE"
        fi
    done
done

{
echo "=========================================="
echo ""
echo "Saved to:"
echo "  Text : $RESULTS_FILE"
echo "  CSV  : $RESULTS_CSV"
echo "=========================================="
} | tee -a "$RESULTS_FILE"

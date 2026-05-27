# Experiment Schedule (commands to run inside salloc)

> **You** request the GPUs with `salloc` and run the commands below.
> I won't `sbatch` anything.

## GPU allocation

Available idle (from your earlier `sinfo`):

| Partition / Node | GPU | Best for |
|---|---|---|
| `nova21-fro` | `a40:4` (4 × 48 GB) | **best** — IQL sweep + parallel diffusion |
| `nova24-gh-` | `gh200:1` (96 GB) | one huge diffusion run |
| `nova20-fro` | `rtx_2080`, `rtx_6000` (24 GB) | eval, single-GPU runs |

### Recommended salloc commands

```bash
# Phase A — eval-only (need ~30 min, 1 GPU): cheapest
salloc -A mech-ai --gres=gpu:rtx_6000:1 -c 8 --mem=32G -t 2:00:00

# Phase B — full sweep (~24 h, 4 GPUs): everything
salloc -A mech-ai --gres=gpu:a40:4 -c 16 --mem=64G -t 24:00:00

# Phase C — one diffusion retrain (~12 h, 1 big GPU)
salloc -A mech-ai --gres=gpu:gh200:1 -c 16 --mem=64G -t 12:00:00
```

---

## Phase 0 — One-time setup (5 min)

Inside the salloc shell:

```bash
cd /work/mech-ai-scratch/supersai/disa-rl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/supersai/.conda/envs/disa

# Install missing eval dependencies
pip install gymnasium "mujoco<3" imageio

# Sanity
python -c "import gymnasium, mujoco; print('OK')"
```

---

## Phase 1 — Evaluate existing checkpoints (~30 min)

We already have 660+ trained IQL checkpoints across 4 envs × 3 alphas × 5 seeds (medium-v2)
plus 3 seeds × 1 alpha (medium-replay-v2). They were never scored because gymnasium
wasn't installed on the cluster.

```bash
# Fast pass: only final.pt per run → one CSV
python eval_final_only.py --n_episodes 10 --out results/eval_finals.csv
```

This writes:
- `results/eval_finals.csv`           per-run scores
- `results/eval_finals_summary.csv`   mean ± std per (env, mode, alpha)

Expected output: many rows with very low scores on medium-v2 (because the syn data
that fed those runs was mode-collapsed) and decent scores on medium-replay-v2.
This confirms our diagnosis.

---

## Phase 2 — Regenerate synthetic data with the bug fix (~2 GPU·hr × 4 envs in parallel)

The `target_return` bug is fixed; this regenerates clean synthetic data.

```bash
# Run 4 envs in parallel on a 4-GPU node
for i in 0 1 2 3; do
  ENV=(halfcheetah hopper walker2d ant)
  CUDA_VISIBLE_DEVICES=$i python generate_synthetic_data.py \
      --env "${ENV[$i]}-medium-v2" \
      --return_sampling topk \
      --cfg_scale 1.2 \
      > "logs/gen_${ENV[$i]}-medium-v2.log" 2>&1 &
done
wait

# Quick sanity check — synthetic reward should now span a realistic range
python -c "
import numpy as np
for env in ['halfcheetah','hopper','walker2d','ant']:
    d = np.load(f'data/synthetic/{env}-medium-v2/synthetic_transitions.npz')
    r = d['rewards']; o = d['observations']
    print(f'{env:<12} r=[{r.min():.2f},{r.max():.2f}] m={r.mean():.2f} s={r.std():.2f}  obs_std={o.std():.2f}')
"
```

**Decision point.** If the new syn data has reasonable reward distribution
(std > 0.5, mean within real range), continue to Phase 3.
If still degenerate, retrain diffusion (Phase 2b).

### Phase 2b — Retrain diffusion (only if Phase 2 syn data is still bad)

```bash
# Smaller model (5M vs 39M params) — defaults are now correct in train.py
# 12-24 hours per env. Recommend GH200 for one env, or A40 in parallel.
CUDA_VISIBLE_DEVICES=0 python diffusion/train.py \
    --env halfcheetah-medium-v2 \
    --hidden_size 256 --depth 6 --num_heads 4 --mlp_dropout 0.3 \
    --num_steps 200000 --batch_size 256 \
    > logs/diffusion_halfcheetah_v2.log 2>&1
```

---

## Phase 3 — Sanity-check augmented IQL (~6 h on 4 GPUs)

Re-run IQL `alpha=0.5` on each of the 4 envs, *one seed only* first, to confirm
the new syn data improves over the offline_only baseline. With eval working,
WandB will now log proper normalized scores.

```bash
for i in 0 1 2 3; do
  ENV=(halfcheetah hopper walker2d ant)
  CUDA_VISIBLE_DEVICES=$i python iql/train_iql.py \
      --env "${ENV[$i]}-medium-v2" \
      --mode augmented --alpha 0.5 --bc_weight 0.1 --seed 0 \
      --num_steps 500000 \
      --wandb_project disa-rl-medium-fixed \
      > "logs/sanity_${ENV[$i]}-medium-v2_a0.5_s0.log" 2>&1 &
done
wait
```

After this, eyeball the WandB curves and the last lines:

```bash
for env in halfcheetah hopper walker2d ant; do
  echo "=== $env ==="
  grep "normalized=" logs/sanity_${env}-medium-v2_a0.5_s0.log | tail -3
done
```

**Decision point.** If at least 2 of 4 envs show > 5 points improvement over
the published IQL baseline, run the full sweep (Phase 4). Otherwise jump to VCDG (Phase 5).

---

## Phase 4 — Full alpha × seed sweep (~24 h on 4 GPUs)

This is the same workload your `slurm_iql_medium.sh` already does. With eval
now working, you'll get real scores in WandB.

```bash
# 4 envs × 3 alphas × 5 seeds = 60 runs at 1M steps
# On a40:4, each GPU handles 1 env at a time, 5 seeds concurrently per env per alpha
bash scripts/run_iql_sweep_local.sh   # see scripts/ for ready-made version
```

(I haven't created `run_iql_sweep_local.sh` yet — if you want this exact
sweep on salloc instead of sbatch, ping me and I'll generate it.)

---

## Phase 4.5 — v2 sweep with PARS-adopted reward scaling + PA loss + alpha warmup

Added during the literature scan (2026-05-12). Full command in
`scripts/launch_v2_sweep.sh`. Run AFTER the v1 sweep finishes:

```bash
bash scripts/launch_v2_sweep.sh
```

Launches 12 jobs (4 envs × 3 method variants: DRC-medium, baseline-medium, DRC-replay)
with env-specific reward scaling (5 or 10) and PA loss enabled. ETA ~5-6h.

Compare v1 vs v2 to isolate the contribution of PARS adoption.

## Phase 5 — Run VCDG (novel algorithm) (~12 h on 4 GPUs)

This is the paper-worthy run. Requires Phase 1 critics already trained.

### Step A: Train pure-real IQL critics (~6 h on 4 GPUs)
*(skip if Phase 1 already produced `offline_only` checkpoints with eval scores)*

```bash
for i in 0 1 2 3; do
  ENV=(halfcheetah hopper walker2d ant)
  CUDA_VISIBLE_DEVICES=$i python iql/train_iql.py \
      --env "${ENV[$i]}-medium-v2" \
      --mode offline_only --bc_weight 0.0 --seed 0 \
      --num_steps 1000000 \
      --wandb_project disa-rl-medium-fixed \
      > "logs/offline_${ENV[$i]}.log" 2>&1 &
done
wait
```

### Step B: Generate VCDG synthetic data (~2 h on 4 GPUs)

```bash
for i in 0 1 2 3; do
  ENV=(halfcheetah hopper walker2d ant)
  CUDA_VISIBLE_DEVICES=$i python generate_synthetic_data.py \
      --env "${ENV[$i]}-medium-v2" \
      --return_sampling topk --cfg_scale 1.0 \
      --vcdg --vcdg_guidance_scale 0.5 \
      > "logs/vcdg_gen_${ENV[$i]}.log" 2>&1 &
done
wait
```

Outputs `data/synthetic/<env>/synthetic_transitions_vcdg.npz`.

### Step C: Train IQL on VCDG-augmented data (~6 h on 4 GPUs, 5 seeds × 4 envs)

```bash
for i in 0 1 2 3; do
  ENV=(halfcheetah hopper walker2d ant)
  for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$i python iql/train_iql.py \
        --env "${ENV[$i]}-medium-v2" \
        --mode augmented --alpha 0.5 --bc_weight 0.1 --seed $seed \
        --use_vcdg_data \
        --num_steps 1000000 \
        --wandb_project disa-rl-vcdg \
        > "logs/vcdg_iql_${ENV[$i]}_s${seed}.log" 2>&1 &
  done
done
wait
```

Then `python eval_final_only.py` to summarize.

---

## Ablation grid (for the paper)

| # | Method | flag(s) | What it isolates |
|---|---|---|---|
| 1 | IQL (real only) | `--mode offline_only` | baseline |
| 2 | IQL + naive syn | `--mode augmented` | shows OOD problem |
| 3 | IQL + clean syn  (bug-fix only) | `--mode augmented` after Phase 2 | value of correct target_return |
| 4 | IQL + VCDG syn | `--mode augmented --use_vcdg_data` | full method |
| 5 | VCDG, no Q-anomaly | `--vcdg --no_vcdg_q_anomaly` | isolates filter contribution |
| 6 | VCDG, no TD-relabel | `--vcdg --no_vcdg_td_relabel` | isolates reward contribution |
| 7 | VCDG, guidance=0 | `--vcdg --vcdg_guidance_scale 0` | isolates guidance contribution |

5 seeds each, 4 envs (medium-v2 first; replay-v2 later) → 7 × 4 × 5 = 140 runs.

---

## Things I changed in this commit

1. **`generate_synthetic_data.py`** — fixed `target_return` (used episode return; now uses sub-traj p90 + per-batch sampling); added `--vcdg` end-to-end.
2. **`diffusion/value_guided.py`** — new file with `value_guided_heun`, `td_relabel_rewards`, `q_anomaly_mask`.
3. **`diffusion/generate.py`** — `TrajectoryGenerator.generate` now accepts a `value_fn` and dispatches to the value-guided sampler.
4. **`diffusion/train.py`** — exposed `--hidden_size`, `--depth`, `--num_heads`, `--mlp_dropout`, `--weight_decay`; defaults are now the small (5M) model.
5. **`iql/train_iql.py`** — `--use_vcdg_data` flag.
6. **`eval_final_only.py`** — new, walks all checkpoints, writes CSV.
7. **`scripts/eval_all_checkpoints.sh`** — convenience wrapper.
8. **`legacy/`** — moved 16 redundant scripts (`run_disa_rl.py`, `slurm_iql.sh`, etc).
9. **`RESEARCH_PLAN.md`** — paper-direction doc with the novelty pitch.

"""
Parse training logs and produce a paper-table CSV.

For every (env, mode, alpha, seed) found in the logs, extract:
    - best normalized score during training
    - final normalized score (last eval tick)
    - step of best score
    - count of eval ticks observed

Writes:
    results/agg_table.csv         (per-run rows)
    results/agg_summary.csv       (mean ± std per cell)

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --logs_glob "logs/v2_*.log"
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict


_NORM_RE = re.compile(r"\[\s*(\d+)\]\s+normalized=\s*([\d.+-]+)\s+return=\s*([\d.+-]+)")
_ENV_RE  = re.compile(r"(halfcheetah|hopper|walker2d|ant)-(medium|medium-replay|expert)-v2")


def parse_log(path: str) -> dict:
    """Return dict with per-run metadata + score history."""
    name = os.path.basename(path)
    env_m = _ENV_RE.search(name)
    env = env_m.group(0) if env_m else "unknown"

    # tag = whatever prefix is before the env. e.g. "v1_", "base_", "v1bipedal_"
    tag_part = name.split(env)[0].rstrip("_")
    seed_m = re.search(r"_s(\d+)", name)
    seed = int(seed_m.group(1)) if seed_m else -1
    alpha_m = re.search(r"_a([0-9.]+)", name)
    alpha = float(alpha_m.group(1)) if alpha_m else float("nan")

    scores = []
    with open(path, errors="replace") as f:
        for line in f:
            m = _NORM_RE.search(line)
            if m:
                step  = int(m.group(1))
                norm  = float(m.group(2))
                rtn   = float(m.group(3))
                scores.append((step, norm, rtn))

    if not scores:
        return {"tag": tag_part, "env": env, "seed": seed, "alpha": alpha,
                "n_evals": 0, "best": None, "final": None, "best_step": None,
                "log_path": path}

    best = max(scores, key=lambda x: x[1])
    final = scores[-1]
    return {
        "tag": tag_part, "env": env, "seed": seed, "alpha": alpha,
        "n_evals": len(scores),
        "best":     best[1], "best_step": best[0],
        "final":    final[1], "final_step": final[0],
        "log_path": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_glob", type=str, default="logs/*.log",
                    help="Glob pattern (default: all logs/*.log)")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(args.logs_glob)):
        # Skip non-iql logs (gen, idm, eval are different)
        b = os.path.basename(path)
        if b.startswith(("gen_", "idm_", "eval_", "diffusion_")):
            continue
        r = parse_log(path)
        if r["n_evals"] > 0:
            rows.append(r)

    if not rows:
        print(f"No scored logs matching {args.logs_glob}.")
        return

    # Per-run table
    table_path = os.path.join(args.out_dir, "agg_table.csv")
    with open(table_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "env", "alpha", "seed",
                                           "n_evals", "best", "best_step",
                                           "final", "final_step", "log_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {table_path}")

    # Summary: mean ± std over seeds per (tag, env, alpha)
    summary_path = os.path.join(args.out_dir, "agg_summary.csv")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["tag"], r["env"], r["alpha"])].append((r["best"], r["final"]))

    with open(summary_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["tag", "env", "alpha", "n_seeds",
                    "best_mean", "best_std", "final_mean", "final_std"])
        for key, scores in sorted(cells.items()):
            bests  = [s[0] for s in scores]
            finals = [s[1] for s in scores]
            n = len(scores)
            bm, bs = (st.mean(bests),  st.pstdev(bests)  if n > 1 else 0.0)
            fm, fs = (st.mean(finals), st.pstdev(finals) if n > 1 else 0.0)
            w.writerow([key[0], key[1], f"{key[2]:.2f}", n,
                        f"{bm:.2f}", f"{bs:.2f}", f"{fm:.2f}", f"{fs:.2f}"])
    print(f"Wrote summary to {summary_path}")

    # Print summary to stdout for quick inspection
    print()
    print(f"{'tag':<14} {'env':<32} {'alpha':<6} {'n':<3} {'best':<16} {'final':<16}")
    print("-" * 96)
    for key, scores in sorted(cells.items()):
        bests  = [s[0] for s in scores]
        finals = [s[1] for s in scores]
        n = len(scores)
        bm, bs = (st.mean(bests),  st.pstdev(bests)  if n > 1 else 0.0)
        fm, fs = (st.mean(finals), st.pstdev(finals) if n > 1 else 0.0)
        print(f"{key[0]:<14} {key[1]:<32} {key[2]:<6.2f} {n:<3} "
              f"{bm:6.1f} ± {bs:5.1f}   {fm:6.1f} ± {fs:5.1f}")


if __name__ == "__main__":
    main()

"""
Publication-grade aggregator for DiSA-RL run logs.

For each input log file, extract every `normalized=X.X` printed by
`iql/train_iql.py`. Group by (env, method) where:
    env    = halfcheetah | hopper | walker2d | ant      (from filename)
    method = everything between the env and the seed tag in the filename

For each group, compute:
    - last-K-avg per seed  (default K=10; honest protocol per
      [[project-eval-methodology]] — no running-max)
    - mean ± std across seeds
    - max step reached (so the reader can see whether the runs converged)

Prints a Markdown table. Optionally writes results/agg_*.csv.

Examples:
    python scripts/report_runs.py 'logs/s2_*.log' 'logs/cmp_*.log'
    python scripts/report_runs.py 'logs/td3bc_*.log' --last 20
    python scripts/report_runs.py 'logs/s2p5_*.log' --csv results/s2p5.csv
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


_ENVS = ("halfcheetah", "hopper", "walker2d", "ant")
_NORM_RE = re.compile(r"normalized=\s*([-\d.]+)")
_STEP_RE = re.compile(r"(\d+)/[0-9]+")
_SEED_RE = re.compile(r"_s(\d+)")


def _split_name(name: str) -> tuple[str, str, int]:
    """Return (env, method_tag, seed) from a log basename.

    method_tag is everything between the leading "<exp>_<env>_" and the
    trailing "_s<seed>"; "_<env>" is also stripped if present.
    """
    seed = -1
    m = _SEED_RE.search(name)
    if m:
        seed = int(m.group(1))
    env = "?"
    for e in _ENVS:
        if e in name:
            env = e
            break

    stem = name.replace(".log", "")
    # Drop "_s<seed>"
    stem = _SEED_RE.sub("", stem)
    # Drop "_<env>" if present
    for e in _ENVS:
        stem = re.sub(rf"_?{e}(-medium-replay-v2|-medium-v2|-medium-expert-v2)?_?",
                      "_", stem).strip("_")
    method = stem if stem else "?"
    return env, method, seed


def parse_one(path: str, last_k: int = 10) -> dict | None:
    """Parse a single log; return a dict per (env, method, seed)."""
    txt = open(path, errors="replace").read()
    norms = [float(x) for x in _NORM_RE.findall(txt)]
    if not norms:
        return None
    last = float(np.mean(norms[-last_k:]))
    steps = _STEP_RE.findall(txt)
    max_step = int(steps[-1]) if steps else 0
    env, method, seed = _split_name(os.path.basename(path))
    return dict(env=env, method=method, seed=seed,
                last_k_avg=last, max_step=max_step, n_evals=len(norms),
                path=path)


def aggregate(rows: list[dict]) -> list[dict]:
    """Group by (env, method); return one summary row per group."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["env"], r["method"])].append(r)
    out = []
    for (env, method), members in sorted(groups.items()):
        vals = [m["last_k_avg"] for m in members]
        steps = [m["max_step"] for m in members]
        out.append(dict(
            env=env, method=method,
            n=len(vals),
            seeds=sorted(m["seed"] for m in members),
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            min=float(np.min(vals)),
            max=float(np.max(vals)),
            min_step=int(min(steps)),
            max_step=int(max(steps)),
        ))
    return out


def fmt_md(summary: list[dict], last_k: int) -> str:
    """Markdown table."""
    lines = [
        f"| env | method | n | last{last_k}avg mean±std | per-seed | step range |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(summary, key=lambda x: (x["env"], x["method"])):
        ms = f"{r['mean']:.2f} ± {r['std']:.2f}"
        rng = f"{r['min_step']:,}..{r['max_step']:,}"
        # find per-seed values from rows shouldn't include here — just min/max
        per = f"[{r['min']:.1f}, {r['max']:.1f}]"
        lines.append(f"| {r['env']} | {r['method']} | {r['n']} | {ms} | {per} | {rng} |")
    return "\n".join(lines)


def write_csv(rows: list[dict], summary: list[dict], path: str) -> None:
    import csv as _csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        w = _csv.writer(f)
        w.writerow(["scope", "env", "method", "seed_or_n",
                    "last_k_avg_or_mean", "std", "min", "max",
                    "min_step", "max_step"])
        for r in sorted(rows, key=lambda x: (x["env"], x["method"], x["seed"])):
            w.writerow(["per-run", r["env"], r["method"], r["seed"],
                        f"{r['last_k_avg']:.4f}", "", "", "",
                        "", r["max_step"]])
        for s in sorted(summary, key=lambda x: (x["env"], x["method"])):
            w.writerow(["summary", s["env"], s["method"], s["n"],
                        f"{s['mean']:.4f}", f"{s['std']:.4f}",
                        f"{s['min']:.4f}", f"{s['max']:.4f}",
                        s["min_step"], s["max_step"]])
    print(f"wrote {path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("globs", nargs="+",
                   help="log glob patterns, e.g. 'logs/s2_*.log'")
    p.add_argument("--last", type=int, default=10,
                   help="last-K eval ticks to average per run (default 10)")
    p.add_argument("--csv", default=None,
                   help="optional output CSV path")
    p.add_argument("--show_rows", action="store_true",
                   help="also print per-run rows (env, method, seed, last_k_avg, step)")
    args = p.parse_args()

    paths = sorted({pp for g in args.globs for pp in glob.glob(g)})
    if not paths:
        print("no matching log files", file=sys.stderr)
        sys.exit(1)

    rows = [r for r in (parse_one(p, last_k=args.last) for p in paths) if r]
    if not rows:
        print("no normalized= matches in any log", file=sys.stderr)
        sys.exit(1)

    summary = aggregate(rows)

    if args.show_rows:
        print("\n### per-run rows")
        print(f"{'env':<12} {'method':<28} {'seed':>4} {'last_k_avg':>10} {'step':>10}")
        for r in sorted(rows, key=lambda x: (x["env"], x["method"], x["seed"])):
            print(f"{r['env']:<12} {r['method']:<28} {r['seed']:>4} "
                  f"{r['last_k_avg']:>10.2f} {r['max_step']:>10,}")
        print()

    print(f"### last-{args.last}-avg mean±std across seeds")
    print(fmt_md(summary, args.last))
    print()

    if args.csv:
        write_csv(rows, summary, args.csv)


if __name__ == "__main__":
    main()

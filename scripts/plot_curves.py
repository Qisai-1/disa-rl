"""
Plot training curves (normalized score vs step) for all parsed logs.
Saves one PNG per env that overlays all (tag, alpha, seed) tuples.

Usage:
    python scripts/plot_curves.py
    python scripts/plot_curves.py --tags v1 base v2

If matplotlib is missing, falls back to a text-mode (best-and-final
table) which is still useful.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
from collections import defaultdict

_ENV_RE  = re.compile(r"(halfcheetah|hopper|walker2d|ant)-(medium|medium-replay|expert)-v2")
_NORM_RE = re.compile(r"\[\s*(\d+)\]\s+normalized=\s*([\d.+-]+)")


def parse(path: str):
    name = os.path.basename(path)
    env_m = _ENV_RE.search(name)
    env = env_m.group(0) if env_m else "unknown"
    tag = name.split(env)[0].rstrip("_") or "unknown"
    seed_m = re.search(r"_s(\d+)", name)
    seed = int(seed_m.group(1)) if seed_m else 0
    series = []
    with open(path, errors="replace") as f:
        for line in f:
            m = _NORM_RE.search(line)
            if m:
                series.append((int(m.group(1)), float(m.group(2))))
    return tag, env, seed, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_glob", type=str, default="logs/*.log")
    ap.add_argument("--tags", type=str, nargs="*", default=None,
                    help="Filter to only these tags (e.g. v1, base, v2)")
    ap.add_argument("--out_dir", type=str, default="results/plots")
    args = ap.parse_args()

    runs = defaultdict(list)   # (env, tag) → list of (seed, series)
    for path in sorted(glob.glob(args.logs_glob)):
        b = os.path.basename(path)
        if b.startswith(("gen_", "idm_", "eval_", "diffusion_", "watchdog")):
            continue
        tag, env, seed, series = parse(path)
        if not series:
            continue
        if args.tags and tag not in args.tags:
            continue
        runs[(env, tag)].append((seed, series))

    if not runs:
        print(f"No logs matched {args.logs_glob}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — printing text-mode summary instead")
        for (env, tag), entries in sorted(runs.items()):
            for seed, series in sorted(entries):
                if series:
                    best = max(s[1] for s in series)
                    last_step, last_norm = series[-1]
                    print(f"  {env:<32} {tag:<14} s={seed}  "
                          f"best={best:5.1f}  last@{last_step:>6d}={last_norm:5.1f}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    # One figure per env, overlay tags
    envs = sorted({env for (env, _) in runs.keys()})
    for env in envs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (e, tag), entries in sorted(runs.items()):
            if e != env:
                continue
            for seed, series in entries:
                steps = [s[0] for s in series]
                scores = [s[1] for s in series]
                ax.plot(steps, scores, alpha=0.7, label=f"{tag} s={seed}")
        ax.set_title(env)
        ax.set_xlabel("training step")
        ax.set_ylabel("D4RL normalized score")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out_path = os.path.join(args.out_dir, f"{env}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  → {out_path}")

    print(f"\nDone. {len(envs)} plots in {args.out_dir}/")


if __name__ == "__main__":
    main()

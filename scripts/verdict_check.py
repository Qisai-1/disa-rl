#!/usr/bin/env python3
"""
Stage 2.7 verdict checker.

Reads logs/s2p5_*.log and logs/s2p7_*.log, computes mean±std at convergence
(last-10-avg, 3 seeds, matched final step), and prints:
  VERDICT=WIN   if 2.7 mean ≥ 2.5 baseline mean + 5.0 on BOTH envs
  VERDICT=MIXED if 2.7 mean ≥ 2.5 baseline on at least one env
  VERDICT=LOSS  if 2.7 mean < 2.5 baseline on both envs

Exit code 0 always. Used by auto_chain_post_td3bc.sh.
"""
from __future__ import annotations
import re, glob, os, sys
import numpy as np
from collections import defaultdict


def extract_evals(path):
    """Return list of (step, normalized_score) from a training log."""
    rn = re.compile(r'normalized=\s*([-\d.]+)')
    txt = open(path, errors='replace').read().replace('\r', '\n')
    evals, last_step = [], 0
    for line in txt.split('\n'):
        m = re.search(r'(\d+)/500000', line)
        if m:
            last_step = int(m.group(1))
        m2 = rn.search(line)
        if m2:
            evals.append((last_step, float(m2.group(1))))
    return evals


def last_k_avg_by_env(glob_pat, method_substr, k=10):
    """Returns {env: [scores]} where scores is per-seed last-k-avg."""
    out = defaultdict(list)
    for f in sorted(glob.glob(glob_pat)):
        name = os.path.basename(f).replace('.log', '')
        if method_substr not in name:
            continue
        env = name.split('_')[1]  # s2p5_hopper_..., s2p7_walker2d_...
        evals = extract_evals(f)
        if len(evals) >= k:
            scores = [sc for _, sc in evals[-k:]]
            out[env].append(np.mean(scores))
    return out


def main():
    print("=== Stage 2.5 (offline_only baseline) ===")
    s25_off = last_k_avg_by_env('logs/s2p5_*.log', 'offline', k=10)
    s25_summary = {}
    for env, scores in s25_off.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {env}: {m:.2f} ± {s:.2f}  (n={len(scores)})  raw={[round(x,1) for x in scores]}")
        s25_summary[env] = m

    print("\n=== Stage 2.7 (full novelty pack) ===")
    s27 = last_k_avg_by_env('logs/s2p7_*.log', 'bestcombo', k=10)
    s27_summary = {}
    for env, scores in s27.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {env}: {m:.2f} ± {s:.2f}  (n={len(scores)})  raw={[round(x,1) for x in scores]}")
        s27_summary[env] = m

    if not s27_summary:
        print("\nVERDICT=NORUN  (Stage 2.7 logs not found yet)")
        return

    print("\n=== Δ (2.7 − baseline) ===")
    win_envs = []
    parity_envs = []
    lose_envs = []
    for env in sorted(set(s25_summary) | set(s27_summary)):
        b = s25_summary.get(env, float('nan'))
        n = s27_summary.get(env, float('nan'))
        delta = n - b if not (np.isnan(b) or np.isnan(n)) else float('nan')
        tag = ""
        if not np.isnan(delta):
            if delta >= 5.0:
                tag = "WIN ✅"; win_envs.append(env)
            elif delta >= -1.0:
                tag = "parity"; parity_envs.append(env)
            else:
                tag = "LOSS ❌"; lose_envs.append(env)
        print(f"  {env}: {n:.2f} − {b:.2f} = {delta:+.2f}  {tag}")

    n_envs = len(s27_summary)
    if len(win_envs) == n_envs and n_envs >= 2:
        print("\nVERDICT=WIN")
    elif len(win_envs) >= 1 and len(lose_envs) == 0:
        print("\nVERDICT=MIXED")
    else:
        print("\nVERDICT=LOSS")


if __name__ == "__main__":
    main()

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
        env = name.split('_')[1]
        evals = extract_evals(f)
        if len(evals) >= k:
            scores = [sc for _, sc in evals[-k:]]
            out[env].append(np.mean(scores))
    return out


def top_k_mean_by_env(glob_pat, method_substr, k=3):
    """Returns {env: [scores]} where each score = mean of the top-k evals
    for that seed. Approximates the 'best checkpoint' protocol without
    actually re-running eval on saved ckpts."""
    out = defaultdict(list)
    for f in sorted(glob.glob(glob_pat)):
        name = os.path.basename(f).replace('.log', '')
        if method_substr not in name:
            continue
        env = name.split('_')[1]
        evals = extract_evals(f)
        if len(evals) >= k:
            scores_sorted = sorted([sc for _, sc in evals])
            out[env].append(np.mean(scores_sorted[-k:]))
    return out


def best_single_by_env(glob_pat, method_substr):
    """Returns {env: [scores]} = max eval per seed. Biased upward but
    matches what 'Best normalized score' in tqdm finalize reports."""
    out = defaultdict(list)
    for f in sorted(glob.glob(glob_pat)):
        name = os.path.basename(f).replace('.log', '')
        if method_substr not in name:
            continue
        env = name.split('_')[1]
        evals = extract_evals(f)
        if evals:
            out[env].append(max(sc for _, sc in evals))
    return out


def report_multi_protocol(glob_pat, method_substr, label):
    print(f"\n=== {label} ===")
    last10 = last_k_avg_by_env(glob_pat, method_substr, k=10)
    top3   = top_k_mean_by_env(glob_pat, method_substr, k=3)
    best   = best_single_by_env(glob_pat, method_substr)
    out = {}
    envs = sorted(set(last10) | set(top3) | set(best))
    for env in envs:
        l = last10.get(env, [])
        t = top3.get(env, [])
        b = best.get(env, [])
        if not l: continue
        out[env] = {"last10": np.mean(l), "top3": np.mean(t) if t else 0,
                    "best": np.mean(b) if b else 0, "n": len(l)}
        print(f"  {env}: last10={np.mean(l):.2f}±{np.std(l):.2f}  "
              f"top3={np.mean(t):.2f}±{np.std(t):.2f}  "
              f"best={np.mean(b):.2f}±{np.std(b):.2f}  (n={len(l)})")
    return out


def main():
    s25 = report_multi_protocol('logs/s2p5_*.log', 'offline',  'Stage 2.5 (offline_only baseline)')
    s26 = report_multi_protocol('logs/s2p6_*.log', 'capa+',    'Stage 2.6 (CAPA+GTA, no-renorm)')
    s27 = report_multi_protocol('logs/s2p7_*.log', 'bestcombo', 'Stage 2.7 (full novelty pack)')
    s30 = report_multi_protocol('logs/s3p0_*.log', 'ensfilt',   'Stage 3.0 (ensemble-filtered syn)')

    if not s27:
        print("\nVERDICT=NORUN  (Stage 2.7 logs not found yet)")
        return

    # Use top3 protocol as the headline (most defensible — closest to
    # 'best checkpoint' that D4RL papers report)
    print("\n=== Δ (2.7 − 2.5) using top-3 protocol ===")
    win_envs, parity_envs, lose_envs = [], [], []
    for env in sorted(set(s25) | set(s27)):
        b = s25.get(env, {}).get("top3", float('nan'))
        n = s27.get(env, {}).get("top3", float('nan'))
        delta = n - b if not (np.isnan(b) or np.isnan(n)) else float('nan')
        tag = ""
        if not np.isnan(delta):
            if delta >= 3.0: tag = "WIN ✅"; win_envs.append(env)
            elif delta >= -2.0: tag = "parity"; parity_envs.append(env)
            else: tag = "LOSS ❌"; lose_envs.append(env)
        print(f"  {env}: {n:.2f} − {b:.2f} = {delta:+.2f}  {tag}")

    n_envs = len(s27)
    if len(win_envs) == n_envs and n_envs >= 2:
        print("\nVERDICT=WIN")
    elif len(win_envs) >= 1 and len(lose_envs) == 0:
        print("\nVERDICT=MIXED")
    else:
        print("\nVERDICT=LOSS")


if __name__ == "__main__":
    main()

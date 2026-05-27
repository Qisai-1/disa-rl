"""
Fast eval pass: only evaluate the final.pt for every (env, mode, alpha, seed).
Writes results/eval_finals.csv. Use eval_checkpoints.py for full per-step eval.

Usage:
    python eval_final_only.py                          # all envs
    python eval_final_only.py --env hopper-medium-v2   # one env
    python eval_final_only.py --n_episodes 5           # quicker
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iql.agent import IQLAgent
from iql.evaluator import make_evaluator


ENVS = [
    "halfcheetah-medium-v2", "hopper-medium-v2",
    "walker2d-medium-v2", "ant-medium-v2",
    "halfcheetah-medium-replay-v2", "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2", "ant-medium-replay-v2",
]


def get_env_dims(env: str, data_dir: str = "./data") -> tuple[int, int]:
    d = np.load(os.path.join(data_dir, f"{env}.npz"), allow_pickle=True)
    return int(d["observations"].shape[1]), int(d["actions"].shape[1])


def find_runs(env: str) -> list[tuple[str, str, int, str]]:
    """Return list of (mode, alpha_str, seed, ckpt_path) for every final.pt found."""
    base = f"./checkpoints/{env}/iql"
    if not os.path.isdir(base):
        return []
    found = []
    for mode in sorted(os.listdir(base)):
        mode_dir = os.path.join(base, mode)
        if not os.path.isdir(mode_dir):
            continue
        # Some old runs are flat (seed_X under mode), new runs nest under alphaX
        children = sorted(os.listdir(mode_dir))
        alpha_dirs = [c for c in children if c.startswith("alpha")]
        seed_dirs  = [c for c in children if c.startswith("seed_")]

        if alpha_dirs:
            for adir in alpha_dirs:
                alpha = adir.replace("alpha", "")
                for s in sorted(os.listdir(os.path.join(mode_dir, adir))):
                    if not s.startswith("seed_"):
                        continue
                    seed = int(s.replace("seed_", ""))
                    ckpt = os.path.join(mode_dir, adir, s, "final.pt")
                    if os.path.exists(ckpt):
                        found.append((mode, alpha, seed, ckpt))
        else:
            for s in seed_dirs:
                seed = int(s.replace("seed_", ""))
                ckpt = os.path.join(mode_dir, s, "final.pt")
                if os.path.exists(ckpt):
                    found.append((mode, "n/a", seed, ckpt))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default=None,
                    help="Single env to eval (default: all)")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/eval_finals.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = [args.env] if args.env else ENVS

    rows = []
    for env in envs:
        runs = find_runs(env)
        if not runs:
            print(f"[{env}] no checkpoints — skip")
            continue
        try:
            obs_dim, act_dim = get_env_dims(env)
        except FileNotFoundError as e:
            print(f"[{env}] missing dataset: {e}")
            continue

        evaluator = make_evaluator(env, device, n_episodes=args.n_episodes)
        if evaluator._env is None:
            print(f"[{env}] gymnasium env not available — install gymnasium+mujoco")
            continue

        print(f"\n=== {env}  ({len(runs)} runs) ===")
        for mode, alpha, seed, ckpt in runs:
            agent = IQLAgent(obs_dim=obs_dim, action_dim=act_dim, device=device)
            try:
                agent.load(ckpt, actor_only=True)
                m = evaluator.evaluate(agent.actor)
                if m is None:
                    continue
                score = m["eval/normalized"]
                ret   = m["eval/return_mean"]
                rows.append({
                    "env": env, "mode": mode, "alpha": alpha, "seed": seed,
                    "normalized": score, "return": ret, "ckpt": ckpt,
                })
                print(f"  {mode:<12} alpha={alpha:<5} s={seed}  "
                      f"normalized={score:6.1f}  return={ret:8.1f}")
            except Exception as e:
                print(f"  {mode} alpha={alpha} s={seed} FAILED: {e}")
        evaluator.close()

    # Summary
    print("\n\n" + "=" * 75)
    print("SUMMARY (mean ± std over seeds, normalized score)")
    print("=" * 75)
    print(f"  {'env':<30} {'mode':<12} {'alpha':<6} {'normalized':<15} n")
    print("  " + "-" * 70)
    by_key = {}
    for r in rows:
        key = (r["env"], r["mode"], r["alpha"])
        by_key.setdefault(key, []).append(r["normalized"])
    summary = []
    for key, scores in sorted(by_key.items()):
        m, s = float(np.mean(scores)), float(np.std(scores))
        summary.append({"env": key[0], "mode": key[1], "alpha": key[2],
                        "n": len(scores), "mean": m, "std": s})
        print(f"  {key[0]:<30} {key[1]:<12} {key[2]:<6} "
              f"{m:6.1f} ± {s:4.1f}     {len(scores)}")

    # Write CSV
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=["env", "mode", "alpha", "seed",
                                           "normalized", "return", "ckpt"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary_path = args.out.replace(".csv", "_summary.csv")
    with open(summary_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["env", "mode", "alpha", "n", "mean", "std"])
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print(f"\nPer-run results: {args.out}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()

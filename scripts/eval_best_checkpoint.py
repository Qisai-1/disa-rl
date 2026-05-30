#!/usr/bin/env python3
"""
Re-evaluate saved checkpoints with N=100 episodes per checkpoint and
report the best-checkpoint score per seed. Standard D4RL protocol;
removes per-eval variance that drags last-K-avg below the policy's
actual capability.

Output:
  Prints a markdown table of (env, method, seed) → (best_step, score)
  + summary mean±std over seeds.

Usage:
  python scripts/eval_best_checkpoint.py \
      --ckpt_glob 'checkpoints/*/iql/offline_only/alpha0.5/seed_*' \
      --n_episodes 100 \
      --obs_norm     # required if --obs_norm was used during training
"""
from __future__ import annotations
import argparse, os, sys, glob, re
import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from iql.agent import IQLAgent
from iql.buffer import ReplayBuffer
from iql.evaluator import Evaluator


ENV_TO_GYM = {
    "hopper-medium-replay-v2":   ("Hopper-v4",      "hopper-medium-replay-v2"),
    "walker2d-medium-replay-v2": ("Walker2d-v4",    "walker2d-medium-replay-v2"),
    "halfcheetah-medium-replay-v2": ("HalfCheetah-v4", "halfcheetah-medium-replay-v2"),
    "ant-medium-replay-v2":      ("Ant-v4",         "ant-medium-replay-v2"),
}


def env_from_ckpt_path(path: str) -> str:
    """checkpoints/<env>/iql/<method>/<...> → <env>"""
    parts = path.split("/")
    return parts[1]


def get_dims_from_data(env: str):
    data = np.load(f"./data/{env}.npz")
    return data["observations"].shape[1], data["actions"].shape[1]


def eval_one_ckpt(ckpt_path: str, env: str, n_episodes: int, obs_norm: bool,
                  reward_norm: str, device) -> float:
    obs_dim, action_dim = get_dims_from_data(env)
    # Build a real buffer with the same norms so obs_mean/obs_std match
    real_buffer = ReplayBuffer(f"./data/{env}.npz", device,
                                reward_scale=1.0,
                                reward_norm=reward_norm,
                                obs_norm=obs_norm)
    # Build evaluator
    gym_name, dataset_name = ENV_TO_GYM[env]
    obs_m = real_buffer.obs_mean if obs_norm else None
    obs_s = real_buffer.obs_std  if obs_norm else None
    evaluator = Evaluator(gym_name, dataset_name, n_episodes=n_episodes,
                          device=device, obs_mean=obs_m, obs_std=obs_s)
    # Load the actor — IQL/CAPA ckpts have 'actor' state dict
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Spin up a bare agent with matching arch (num_critics inferred from ckpt)
    q_keys = list(ckpt["q"].keys())
    n_critics = max((int(k.split(".")[1]) for k in q_keys if k.startswith("qs.")),
                    default=1) + 1 if any(k.startswith("qs.") for k in q_keys) else 2
    agent = IQLAgent(obs_dim=obs_dim, action_dim=action_dim, device=device,
                     num_critics=n_critics, critic_subset_size=2,
                     q_hidden_dims=(256, 256))
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    metrics = evaluator.evaluate(agent.actor)
    if metrics is None:
        return float("nan")
    return float(metrics["eval/normalized"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", required=True,
                    help="e.g. 'checkpoints/*/iql/offline_only/alpha0.5/seed_*'")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--obs_norm", action="store_true",
                    help="Set this if --obs_norm was used during training.")
    ap.add_argument("--reward_norm", default="corl",
                    help="Set to 'corl' for Stage-2.5+ ckpts; 'none' for older.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Walk seed dirs
    seed_dirs = sorted(glob.glob(args.ckpt_glob))
    print(f"Found {len(seed_dirs)} seed dirs matching {args.ckpt_glob!r}")
    by_env_method = {}
    for sd in seed_dirs:
        env = env_from_ckpt_path(sd)
        ckpts = sorted(glob.glob(os.path.join(sd, "step_*.pt"))
                       + glob.glob(os.path.join(sd, "best.pt"))
                       + glob.glob(os.path.join(sd, "final.pt")))
        if not ckpts:
            continue
        seed = re.search(r"seed_(\d+)", sd).group(1)
        method = "_".join(sd.split("/")[-3:-1])  # e.g. offline_only_alpha0.5

        print(f"\n=== {env} / {method} / seed {seed} ===")
        best_score = -1e9; best_ckpt = None
        for ck in ckpts:
            sc = eval_one_ckpt(ck, env, args.n_episodes, args.obs_norm,
                               args.reward_norm, device)
            tag = os.path.basename(ck)
            print(f"  {tag:<22}  {sc:6.2f}")
            if sc > best_score:
                best_score, best_ckpt = sc, tag

        key = (env, method)
        by_env_method.setdefault(key, []).append((seed, best_score, best_ckpt))

    # Summary
    print("\n\n" + "="*70)
    print("=== Best-checkpoint summary (N={} episodes per eval) ===".format(args.n_episodes))
    print("="*70)
    for (env, method), seed_results in sorted(by_env_method.items()):
        scores = [s for _, s, _ in seed_results]
        print(f"\n{env} / {method}:")
        for seed, sc, ck in sorted(seed_results):
            print(f"  seed {seed}: {sc:6.2f}  ({ck})")
        print(f"  → MEAN: {np.mean(scores):.2f} ± {np.std(scores):.2f}  (n={len(scores)})")


if __name__ == "__main__":
    main()

"""
Evaluate all checkpoints and find the best score during training.
Shows: best score, at which step, and final score.

Usage:
    python eval_checkpoints.py
    python eval_checkpoints.py --env halfcheetah-medium-v2 --alpha 0.5
"""
import torch, sys, numpy as np, os, argparse
sys.path.insert(0, '.')
from iql.agent import IQLAgent
from iql.evaluator import make_evaluator


def get_env_dims(env, data_dir="./data"):
    """Read obs/action dims directly from the dataset .npz (no hardcoding)."""
    d = np.load(os.path.join(data_dir, f"{env}.npz"), allow_pickle=True)
    return int(d["observations"].shape[1]), int(d["actions"].shape[1])

def eval_all(env, obs_dim, act_dim, alpha, mode="augmented", n_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev     = make_evaluator(env, device, n_episodes=n_episodes)

    base = f"./checkpoints/{env}/iql/{mode}/seed_"
    # Also check alpha subdir
    alpha_base = f"./checkpoints/{env}/iql/{mode}/alpha{alpha}/seed_"

    print(f"\n{'='*60}")
    print(f"  {env}  alpha={alpha}")
    print(f"{'='*60}")

    all_best_scores = []
    all_final_scores = []

    for seed in range(5):
        # Try both path formats
        seed_dir = None
        for candidate in [f"{alpha_base}{seed}", f"{base}{seed}"]:
            if os.path.exists(candidate):
                seed_dir = candidate
                break

        if seed_dir is None:
            print(f"  seed={seed}: no checkpoint directory found")
            continue

        # Collect all step checkpoints + final
        ckpts = sorted([
            f for f in os.listdir(seed_dir)
            if f.endswith('.pt') and ('step_' in f or f in ['final.pt', 'best.pt'])
        ])

        if not ckpts:
            print(f"  seed={seed}: no .pt files found")
            continue

        seed_scores = {}
        for ckpt_name in ckpts:
            ckpt_path = os.path.join(seed_dir, ckpt_name)
            try:
                agent = IQLAgent(obs_dim=obs_dim, action_dim=act_dim, device=device)
                agent.load(ckpt_path)
                m = ev.evaluate(agent.actor)
                if m:
                    step = agent.total_steps
                    seed_scores[step] = m["eval/normalized"]
            except Exception as e:
                print(f"  seed={seed} {ckpt_name}: error {e}")
                continue

        if not seed_scores:
            continue

        best_step  = max(seed_scores, key=seed_scores.get)
        best_score = seed_scores[best_step]
        final_step = max(seed_scores.keys())
        final_score = seed_scores[final_step]

        all_best_scores.append(best_score)
        all_final_scores.append(final_score)

        print(f"  seed={seed}:  best={best_score:.1f} @ step {best_step:>7,}  |  final={final_score:.1f}")

    ev.close()

    if all_best_scores:
        print(f"  {'─'*50}")
        print(f"  BEST  mean={np.mean(all_best_scores):.1f} ± {np.std(all_best_scores):.1f}")
        print(f"  FINAL mean={np.mean(all_final_scores):.1f} ± {np.std(all_final_scores):.1f}")

    return all_best_scores, all_final_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   type=str, default=None)
    parser.add_argument("--alpha", type=str, default=None)
    parser.add_argument("--mode",  type=str, default="augmented")
    parser.add_argument("--n_episodes", type=int, default=10)
    args = parser.parse_args()

    # External reference scores (D4RL paper IQL) — display only.
    baselines = {
        "halfcheetah-medium-v2": 47.4,
        "hopper-medium-v2":      66.3,
        "walker2d-medium-v2":    78.3,
        "ant-medium-v2":         86.7,
    }
    all_envs = [
        "halfcheetah-medium-v2", "hopper-medium-v2",
        "walker2d-medium-v2", "ant-medium-v2",
        "halfcheetah-medium-replay-v2", "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2", "ant-medium-replay-v2",
    ]
    alphas = ["0.5", "0.25", "0.0"] if args.alpha is None else [args.alpha]
    envs   = [args.env] if args.env else all_envs

    summary = {}
    for env in envs:
        try:
            obs_dim, act_dim = get_env_dims(env)
        except FileNotFoundError:
            print(f"  {env}: dataset .npz not found — skip")
            continue
        summary[env] = {}
        for alpha in alphas:
            best, final = eval_all(env, obs_dim, act_dim, alpha,
                                   args.mode, args.n_episodes)
            if best:
                summary[env][alpha] = {
                    "best_mean":  np.mean(best),
                    "best_std":   np.std(best),
                    "final_mean": np.mean(final),
                    "final_std":  np.std(final),
                }

    # Print comparison table
    print("\n\n" + "="*75)
    print("  RESULTS SUMMARY")
    print("="*75)
    print(f"  {'Env':<30} {'Alpha':>7} {'Best':>10} {'Final':>10} {'Baseline':>10}")
    print("  " + "─"*65)
    for env, alpha_results in summary.items():
        for alpha, r in alpha_results.items():
            baseline = baselines.get(env, "N/A")
            print(f"  {env:<30} {alpha:>7} "
                  f"{r['best_mean']:>7.1f}±{r['best_std']:.1f} "
                  f"{r['final_mean']:>7.1f}±{r['final_std']:.1f} "
                  f"{str(baseline):>10}")
    print("="*75)

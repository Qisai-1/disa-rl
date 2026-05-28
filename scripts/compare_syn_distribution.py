"""
Deep distributional comparison: synthetic (diffusion) vs real D4RL data.

Goes beyond the marginal audit (validate_synthetic_data.py) to probe the
JOINT / CONDITIONAL structure that drives the TD bootstrap:

  1. Per-dim marginal divergence   — standardized mean shift, std ratio,
                                      KS statistic, 1-Wasserstein (per obs dim).
  2. Covariance structure          — Frobenius distance of corr matrices;
                                      reveals broken cross-dim correlations even
                                      when every marginal looks fine.
  3. Dynamics consistency (hopper) — kinematic residual: Δpos − dt·vel.
                                      Real data obeys this (Euler integration);
                                      if synthetic transitions violate it, the
                                      generated s' does NOT follow from (s,a) →
                                      r + γV(s') is a garbage TD target → the
                                      value function gets poisoned.
  4. Self-loop artifact            — fraction of syn rows with next_obs ≈ obs
                                      (the t==T-1 generation bug).

Usage:
    python scripts/compare_syn_distribution.py --env hopper-medium-replay-v2
    python scripts/compare_syn_distribution.py --all-replay
"""
from __future__ import annotations
import argparse, os
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

REPLAY = ["halfcheetah-medium-replay-v2", "hopper-medium-replay-v2",
          "walker2d-medium-replay-v2", "ant-medium-replay-v2"]

# obs layout for the kinematic check: (n_pos, dt). qpos[1:] then qvel.
# Hopper-v2: 5 pos (z + 4 angles) + 6 vel; the x-velocity (qvel[0]) has no
# matching position in obs, so pos[i] pairs with vel[i+1].  dt = 0.008.
KINEMATICS = {
    "hopper-medium-replay-v2":  dict(n_pos=5, vel_offset=6, dt=0.008),
    "walker2d-medium-replay-v2":dict(n_pos=8, vel_offset=9, dt=0.008),  # 8 pos + 9 vel(=17)
}


def episode_next_obs(obs, terminals, timeouts):
    """Build real (obs, next_obs) by shifting within episodes; return valid mask."""
    done = terminals | timeouts
    valid = ~done[:-1]
    return obs[:-1][valid], obs[1:][valid]


def section(title): print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def compare_env(env, data_dir="./data"):
    real = np.load(os.path.join(data_dir, f"{env}.npz"), allow_pickle=True)
    syn  = np.load(os.path.join(data_dir, "synthetic", env,
                                "synthetic_transitions.npz"), allow_pickle=True)
    r_obs = real["observations"].astype(np.float64)
    s_obs = syn["observations"].astype(np.float64)
    r_rew = real["rewards"].astype(np.float64)
    s_rew = syn["rewards"].astype(np.float64)
    term  = real["terminals"].astype(bool)
    tout  = real.get("timeouts", np.zeros(len(r_rew), bool)).astype(bool)

    section(f"{env}   (real n={len(r_obs):,}, syn n={len(s_obs):,}, dim={r_obs.shape[1]})")

    # ── 1. Per-dim marginal divergence ─────────────────────────────────────
    D = r_obs.shape[1]
    std = r_obs.std(0) + 1e-8
    mean_shift = (s_obs.mean(0) - r_obs.mean(0)) / std         # in real-σ units
    std_ratio  = s_obs.std(0) / std
    # KS + Wasserstein on a subsample (full is slow)
    ridx = np.random.choice(len(r_obs), min(20000, len(r_obs)), replace=False)
    sidx = np.random.choice(len(s_obs), min(20000, len(s_obs)), replace=False)
    ks = np.array([ks_2samp(r_obs[ridx, d], s_obs[sidx, d]).statistic for d in range(D)])
    w  = np.array([wasserstein_distance(r_obs[ridx, d], s_obs[sidx, d]) / std[d] for d in range(D)])

    print("  per-dim obs divergence (worst 6 by KS):")
    print(f"  {'dim':>4} {'|mean shift|σ':>13} {'std ratio':>10} {'KS':>7} {'Wass/σ':>8}")
    order = np.argsort(-ks)
    for d in order[:6]:
        print(f"  {d:>4} {mean_shift[d]:>13.2f} {std_ratio[d]:>10.2f} {ks[d]:>7.3f} {w[d]:>8.2f}")
    print(f"  --- mean over dims: |shift|σ={np.abs(mean_shift).mean():.2f}  "
          f"stdR={std_ratio.mean():.2f}  KS={ks.mean():.3f}  Wass/σ={w.mean():.2f}")

    # ── 2. Correlation-structure distance ──────────────────────────────────
    rc = np.corrcoef(r_obs[ridx].T)
    sc = np.corrcoef(s_obs[sidx].T)
    fro = np.linalg.norm(rc - sc) / np.linalg.norm(rc)
    print(f"\n  corr-matrix relative Frobenius dist: {fro:.3f}  "
          f"(0=identical correlation structure; >0.3 = cross-dim structure differs)")

    # ── 3. Reward marginal ─────────────────────────────────────────────────
    print(f"\n  reward:  real μ={r_rew.mean():.3f} σ={r_rew.std():.3f}   "
          f"syn μ={s_rew.mean():.3f} σ={s_rew.std():.3f}   "
          f"KS={ks_2samp(r_rew[ridx], s_rew[sidx]).statistic:.3f}")

    # ── 4. Dynamics consistency (kinematic residual) ───────────────────────
    if env in KINEMATICS and "next_observations" in syn:
        k = KINEMATICS[env]; npos, voff, dt = k["n_pos"], k["vel_offset"], k["dt"]
        # real
        ro, rno = episode_next_obs(r_obs, term, tout)
        r_resid = (rno[:, :npos] - ro[:, :npos]) - dt * ro[:, voff:voff+npos]
        # syn
        so  = s_obs
        sno = syn["next_observations"].astype(np.float64)
        s_resid = (sno[:, :npos] - so[:, :npos]) - dt * so[:, voff:voff+npos]
        # normalize residual by the per-dim scale of the real Δpos
        scale = (np.abs(rno[:, :npos] - ro[:, :npos]).mean(0) + 1e-8)
        rr = np.abs(r_resid).mean(0) / scale
        sr = np.abs(s_resid).mean(0) / scale
        section_msg = (f"  DYNAMICS CONSISTENCY (Δpos − dt·vel, per pos-dim, "
                       f"normalized by real |Δpos|):")
        print("\n" + section_msg)
        print(f"  {'posdim':>6} {'real resid':>11} {'syn resid':>10} {'syn/real':>9}")
        for d in range(npos):
            ratio = sr[d] / (rr[d] + 1e-8)
            print(f"  {d:>6} {rr[d]:>11.3f} {sr[d]:>10.3f} {ratio:>9.1f}x")
        print(f"  --- mean: real={rr.mean():.3f}  syn={sr.mean():.3f}  "
              f"ratio={sr.mean()/(rr.mean()+1e-8):.1f}x   "
              f"(syn≫real ⇒ generated s' violates physics ⇒ poisoned TD targets)")

        # self-loop artifact
        dpos = np.abs(sno - so).mean(1)
        selfloop = float((dpos < 1e-4).mean())
        print(f"\n  self-loop (next_obs≈obs) fraction: {selfloop*100:.2f}%")
    elif "next_observations" not in syn:
        print("\n  (no next_observations in syn npz — skip dynamics check)")


if __name__ == "__main__":
    np.random.seed(0)
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default=None)
    p.add_argument("--all-replay", action="store_true")
    a = p.parse_args()
    envs = REPLAY if a.all_replay else [a.env]
    for e in envs:
        compare_env(e)

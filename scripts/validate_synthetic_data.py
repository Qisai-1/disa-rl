"""
Audit the diffusion-generated synthetic transitions against the real D4RL data,
quantitatively and qualitatively.

For every env it compares data/synthetic/<env>/synthetic_transitions.npz against
data/<env>.npz on:
  - reward distribution      (mean / std / range  — variance-collapse + optimism)
  - observation coverage     (fraction of syn obs outside the real per-dim range)
  - action validity          (D4RL actions live in [-1, 1]; syn must respect it)
  - transition magnitude     ||s' - s||  (implausible jumps = bad dynamics)
  - corruption               (NaN / Inf)
  - terminal rate

Writes:
  results/syn_audit/summary.csv     per-env quantitative table
  results/syn_audit/<env>.png       reward / Δs / action / PCA panels (if matplotlib)

Usage:
  python scripts/validate_synthetic_data.py
  python scripts/validate_synthetic_data.py --env halfcheetah-medium-v2
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

ENVS = [
    "halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2", "ant-medium-v2",
    "halfcheetah-medium-replay-v2", "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2", "ant-medium-replay-v2",
]


def real_next_obs_delta(obs, terminals, timeouts):
    """Δs = s' - s for real data, derived by shifting obs within episodes."""
    boundary = terminals | timeouts          # episode ends at this index
    valid = ~boundary[:-1]                    # i is valid if i is not an end
    return obs[1:][valid] - obs[:-1][valid]


def stats(x):
    return dict(mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()),
                p1=float(np.percentile(x, 1)), p99=float(np.percentile(x, 99)))


def audit_env(env, data_dir="./data", plot=True, out_dir="results/syn_audit"):
    real_path = os.path.join(data_dir, f"{env}.npz")
    syn_path  = os.path.join(data_dir, "synthetic", env, "synthetic_transitions.npz")
    if not (os.path.exists(real_path) and os.path.exists(syn_path)):
        print(f"  {env}: missing data — skip")
        return None

    real = np.load(real_path, allow_pickle=True)
    syn  = np.load(syn_path,  allow_pickle=True)

    r_obs, r_act, r_rew = real["observations"], real["actions"], real["rewards"]
    r_term = real["terminals"].astype(bool)
    r_tout = real["timeouts"].astype(bool) if "timeouts" in real.files \
             else np.zeros_like(r_term)

    s_obs, s_act, s_rew = syn["observations"], syn["actions"], syn["rewards"]
    s_next = syn["next_observations"]
    s_term = syn["terminals"]

    m = {"env": env, "n_real": len(r_obs), "n_syn": len(s_obs)}

    # ── corruption ─────────────────────────────────────────────────────────
    n_bad = sum(int((~np.isfinite(a)).sum()) for a in (s_obs, s_act, s_rew, s_next))
    m["syn_nan_inf"] = n_bad

    # ── reward ─────────────────────────────────────────────────────────────
    rr, rs = stats(r_rew), stats(s_rew)
    m["rew_real_mean"], m["rew_real_std"] = rr["mean"], rr["std"]
    m["rew_syn_mean"],  m["rew_syn_std"]  = rs["mean"], rs["std"]
    m["rew_std_ratio"] = rs["std"] / (rr["std"] + 1e-9)
    # how far the syn mean sits above the real mean, in real-std units
    m["rew_mean_shift_sigma"] = (rs["mean"] - rr["mean"]) / (rr["std"] + 1e-9)

    # ── observation coverage ───────────────────────────────────────────────
    # Robust per-dim band: real [p0.5, p99.5] ± 10% tolerance. Dimensions whose
    # real range is degenerate (near-constant — e.g. Ant's many contact-force
    # dims that are ~0 in the medium datasets) are EXCLUDED: otherwise any syn
    # noise on them scores as "out of range" and dominates the metric.
    lo = np.percentile(r_obs, 0.5, axis=0)
    hi = np.percentile(r_obs, 99.5, axis=0)
    rng = hi - lo
    nondegen = rng > 1e-5
    tol = 0.10 * rng
    oob = (s_obs < lo - tol) | (s_obs > hi + tol)      # (N, obs_dim) bool
    oob_nd = oob[:, nondegen]
    m["obs_dims"]            = int(s_obs.shape[1])
    m["obs_degenerate_dims"] = int((~nondegen).sum())
    m["obs_oob_elem_frac"] = float(oob_nd.mean()) if nondegen.any() else 0.0
    m["obs_oob_row_frac"]  = float(oob_nd.any(1).mean()) if nondegen.any() else 0.0

    # ── action validity (D4RL actions ∈ [-1, 1]) ───────────────────────────
    tol = 1e-3
    m["act_oob_frac"]  = float((np.abs(s_act) > 1.0 + tol).mean())
    m["act_real_oob"]  = float((np.abs(r_act) > 1.0 + tol).mean())
    m["act_abs_max"]   = float(np.abs(s_act).max())

    # ── transition magnitude  ||s' - s|| ───────────────────────────────────
    r_ds = real_next_obs_delta(r_obs, r_term, r_tout)
    s_ds = s_next - s_obs
    r_dn, s_dn = np.linalg.norm(r_ds, axis=1), np.linalg.norm(s_ds, axis=1)
    m["ds_real_mean"], m["ds_syn_mean"] = float(r_dn.mean()), float(s_dn.mean())
    m["ds_std_ratio"] = float(s_dn.std() / (r_dn.std() + 1e-9))
    m["ds_p99_ratio"] = float(np.percentile(s_dn, 99) /
                              (np.percentile(r_dn, 99) + 1e-9))

    # ── terminal rate ──────────────────────────────────────────────────────
    m["term_real"] = float(r_term.mean())
    m["term_syn"]  = float((s_term > 0.5).mean())

    # ── verdict ────────────────────────────────────────────────────────────
    flags = []
    if m["syn_nan_inf"] > 0:                       flags.append("CORRUPT(NaN/Inf)")
    if m["rew_std_ratio"] < 0.5:                   flags.append("reward-variance-collapse")
    if m["rew_mean_shift_sigma"] > 0.5:            flags.append("reward-optimism")
    if m["obs_oob_row_frac"] > 0.20:               flags.append("obs-coverage-drift")
    if m["act_oob_frac"] > 0.01:                   flags.append("action-bound-violation")
    if not (0.5 <= m["ds_std_ratio"] <= 2.0):      flags.append("transition-magnitude-mismatch")
    serious = {"CORRUPT(NaN/Inf)", "action-bound-violation", "obs-coverage-drift"}
    if any(f in serious for f in flags):    m["verdict"] = "SERIOUS"
    elif flags:                            m["verdict"] = "MINOR"
    else:                                  m["verdict"] = "OK"
    m["flags"] = ";".join(flags) if flags else "-"

    # ── plots ──────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(out_dir, exist_ok=True)
            fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))

            ax[0].hist(r_rew, bins=80, alpha=0.55, density=True, label="real")
            ax[0].hist(s_rew, bins=80, alpha=0.55, density=True, label="syn")
            ax[0].set_title(f"{env}\nreward"); ax[0].legend()

            ax[1].hist(r_dn, bins=80, alpha=0.55, density=True, label="real")
            ax[1].hist(s_dn, bins=80, alpha=0.55, density=True, label="syn")
            ax[1].set_title("||s' - s||"); ax[1].legend()

            ax[2].hist(r_act.ravel(), bins=80, alpha=0.55, density=True, label="real")
            ax[2].hist(s_act.ravel(), bins=80, alpha=0.55, density=True, label="syn")
            ax[2].axvline(-1, color="k", ls=":"); ax[2].axvline(1, color="k", ls=":")
            ax[2].set_title("action values"); ax[2].legend()

            # PCA: fit on real obs, project both — shows manifold drift
            k = min(40000, len(r_obs), len(s_obs))
            ri = np.random.default_rng(0).choice(len(r_obs), k, replace=False)
            si = np.random.default_rng(1).choice(len(s_obs), k, replace=False)
            rc = r_obs[ri] - r_obs[ri].mean(0)
            _, _, vt = np.linalg.svd(rc, full_matrices=False)
            comp = vt[:2].T
            rp = (r_obs[ri] - r_obs[ri].mean(0)) @ comp
            sp = (s_obs[si] - r_obs[ri].mean(0)) @ comp
            ax[3].scatter(rp[:, 0], rp[:, 1], s=2, alpha=0.25, label="real")
            ax[3].scatter(sp[:, 0], sp[:, 1], s=2, alpha=0.25, label="syn")
            ax[3].set_title("obs PCA (real-fit)"); ax[3].legend()

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{env}.png"), dpi=100)
            plt.close(fig)
        except ImportError:
            pass

    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="results/syn_audit")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    envs = [args.env] if args.env else ENVS
    rows = []
    for env in envs:
        print(f"auditing {env} ...")
        r = audit_env(env, args.data_dir, plot=not args.no_plot, out_dir=args.out_dir)
        if r:
            rows.append(r)

    if not rows:
        print("no data audited.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # console report
    print("\n" + "=" * 110)
    print("  SYNTHETIC DATA AUDIT")
    print("=" * 110)
    h = (f"  {'env':<30}{'rew σ':>8}{'rew Δμ':>9}{'obs OOB':>9}"
         f"{'act OOB':>9}{'Δs σ':>8}{'verdict':>10}  flags")
    print(h); print("  " + "-" * 106)
    for m in rows:
        print(f"  {m['env']:<30}"
              f"{m['rew_std_ratio']:>8.2f}"
              f"{m['rew_mean_shift_sigma']:>9.2f}"
              f"{m['obs_oob_row_frac']*100:>8.1f}%"
              f"{m['act_oob_frac']*100:>8.2f}%"
              f"{m['ds_std_ratio']:>8.2f}"
              f"{m['verdict']:>10}  {m['flags']}")
    print("=" * 110)
    print("  rew σ      = syn reward std / real reward std        (1.0 = match; <0.5 = collapsed)")
    print("  rew Δμ     = (syn mean - real mean) / real std       (>0.5 = optimistic)")
    print("  obs OOB    = % syn obs rows with >=1 dim outside the real per-dim range")
    print("  act OOB    = % syn action entries outside [-1, 1]")
    print("  Δs σ       = syn ||s'-s|| std / real ||s'-s|| std    (in-band 0.5-2.0)")
    print(f"\n  table → {csv_path}")
    if not args.no_plot:
        print(f"  panels → {args.out_dir}/<env>.png")


if __name__ == "__main__":
    main()

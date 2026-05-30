"""
Value-Calibrated Diffusion Generation (VCDG).

Uses a pretrained IQL value function V_phi to guide flow-matching sampling
toward high-value, in-distribution trajectories.

Algorithm sketch:
    For each Heun step:
        v_uncond = v_theta(x_t, t, c)          # flow-matching prediction
        g        = grad_x [V_phi(s_t)]         # value-function gradient
        v        = v_uncond + lambda_v(t) * g  # guided velocity
        x_{t+1}  = Heun_step(x_t, v, dt)

The trajectory generated this way is then post-processed by:
  - TD-relabel: replace generated reward with V(s) - gamma * V(s')
  - Q-anomaly reject: drop transitions where |Q(s,a) - V(s)| is far above
    the 99th percentile of the same quantity on real data.

This file is independent of `flow_matching.py`. It expects a `cfm` object
that exposes `_velocity(x, t, cond, cfg_scale)` (the existing API).
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple


def value_guided_heun(
    cfm,
    batch_size:    int,
    cond:          Tensor,
    value_fn:      Callable[[Tensor], Tensor],
    obs_dim:       int,
    obs_denorm:    Callable[[np.ndarray], np.ndarray],
    nfe:           int   = 20,
    cfg_scale:     float = 1.5,
    guidance_scale: float = 0.5,
    guidance_schedule: str = "linear-decay",
) -> Tensor:
    """
    Heun sampler with classifier guidance from a state value function.

    Parameters
    ----------
    cfm            : ConditionalFlowMatching object (provides _velocity)
    batch_size     : number of trajectories
    cond           : (B, cond_dim) conditioning tensor
    value_fn       : callable V(x_obs) → (B*T,). Should ACCEPT the obs
                     vector in whatever space the caller chose to expose
                     (typically a closure that internally denormalises
                     before calling the IQL V_phi — see generate.py:_vf_normspace).
                     The gradient flows back through value_fn into x_g.
    obs_dim        : observation dimension
    obs_denorm     : DEPRECATED — ignored; kept only for caller backward-
                     compat. value_fn is responsible for the obs-space
                     transform.
    guidance_scale : λ_v in the formula above
    guidance_schedule : "constant" or "linear-ramp" (alias: "linear-decay"
                       — kept for backward compat). Both ramp UP with t:
                       λ(t) = guidance_scale * t. At t→0 (pure noise) the
                       value gradient is meaningless, so λ≈0; at t→1
                       (near-clean data) λ≈guidance_scale so guidance
                       actually steers the trajectory. Fixed 2026-05-26
                       from a previous (1-t) version that did the opposite.
    """
    device = cfm.device
    T = cfm.model.T
    D = cfm.model.D
    dt = 1.0 / nfe

    # Start from noise. (NORMALIZED space.)
    x = torch.randn((batch_size, T, D), device=device)

    # We need autograd on `x` for the guidance term, so we run the value
    # gradient inside an enable_grad block while the flow-matching pass
    # itself stays in the existing no_grad context.
    for i in range(nfe):
        t_curr_scalar = i * dt
        t_next_scalar = min((i + 1) * dt, 1.0)
        t_curr = torch.full((batch_size,), t_curr_scalar, device=device)
        t_next = torch.full((batch_size,), t_next_scalar, device=device)

        # ── 1. Unconditional / CFG flow velocity ─────────────────────────
        with torch.no_grad():
            v_uncond = cfm._velocity(x, t_curr, cond, cfg_scale)

        # ── 2. Value-function gradient w.r.t. the OBS portion of x ──────
        # Ramp guidance UP with t. At t≈0 the diffusion state is pure
        # noise → V's gradient is meaningless → λ≈0 (no guidance).
        # At t≈1 the state is near-clean data → V's gradient is meaningful
        # → λ≈guidance_scale (full guidance). Bugfix 2026-05-26: the
        # previous formulation (1 - t_curr) inverted this and effectively
        # applied guidance only on noise, which was useless.
        if guidance_schedule in ("linear-ramp", "linear-decay"):
            lam = guidance_scale * t_curr_scalar
        else:
            lam = guidance_scale

        if lam > 0:
            x_g = x.detach().requires_grad_(True)
            with torch.enable_grad():
                # Flatten (B, T, obs_dim) → (B*T, obs_dim), call V, mean-reduce.
                # Mean reduction lets us back-prop a scalar; we keep per-time
                # gradient signal because each x[b, t, :obs_dim] contributes
                # independently to the sum.
                obs_flat = x_g[..., :obs_dim].reshape(-1, obs_dim)
                v_vals   = value_fn(obs_flat)              # (B*T,)
                # Gradient ascent on V — flow target is (x1 - x0), so we
                # add to the velocity in the direction that increases V.
                v_vals.sum().backward()
            grad = x_g.grad.detach()                       # (B, T, D)
            # Apply guidance only to the obs sub-vector
            grad_obs = grad[..., :obs_dim]
            # Normalise grad to unit length per (b, t) to stabilise scale
            grad_norm = grad_obs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_obs = grad_obs / grad_norm
            guidance = torch.zeros_like(v_uncond)
            guidance[..., :obs_dim] = lam * grad_obs
            v_uncond = v_uncond + guidance

        # ── 3. Heun (predictor-corrector) update ────────────────────────
        with torch.no_grad():
            x_pred = x + v_uncond * dt
            v_next = cfm._velocity(x_pred, t_next, cond, cfg_scale)
            x      = x + 0.5 * (v_uncond + v_next) * dt

    return x


def q_guided_partial(
    cfm,
    x1_t:               Tensor,      # (B, T, D) NORMALIZED clean seed
    cond:               Tensor,
    q_fn:               Callable[[Tensor, Tensor], Tensor],  # Q(obs, action) → (N,)
    obs_dim:            int,
    action_dim:         int,
    noise_ratio:        float = 0.5,
    nfe:                int   = 20,
    cfg_scale:          float = 1.5,
    guidance_scale:     float = 0.5,
    guidance_schedule:  str   = "linear-ramp",
) -> Tensor:
    """
    Q-guided variant of GTA's partial-noising sampler.

    Composes:
      - GTA partial-noising (start from x1_t = real clean traj, add noise to
        ratio μ, denoise) — preserves stitching from in-distribution seeds.
      - Q-function guidance on the obs+action channels at each denoising step:
        gradient ascent on Q(s, a) steers the trajectory toward higher-Q
        compositions the diffusion alone might not reach.

    Difference vs value_guided_heun:
      - V-guidance only sees s, so it can only shape obs trajectories; the
        action sub-vector is left to drift with the prior.
      - Q-guidance sees (s, a), so it shapes BOTH obs and action channels.
        Stronger signal, especially for offline-RL since Q is what the
        downstream IQL/CAPA agent actually optimizes — the diffusion samples
        line up with the policy-improvement direction.

    Returns the generated (B, T, D) normalized trajectory.

    Notes:
      - q_fn is expected to take denormalised (s, a) — pass a closure that
        denormalises x_g internally before calling the IQL Q ensemble.
        Use Q.min(s, a) (twin-Q min) for a stable per-row scalar.
      - This mirrors GTA's partial path; for "pure" Q-guided generation from
        noise, drop the partial-noising seed and start from torch.randn.
    """
    cfm_model = cfm.model
    device = cfm.device
    T = cfm_model.T
    D = cfm_model.D
    assert x1_t.shape == (x1_t.shape[0], T, D), f"x1_t {x1_t.shape} ≠ ({x1_t.shape[0]}, {T}, {D})"
    assert D >= obs_dim + action_dim, f"D={D} < obs+action={obs_dim+action_dim}"

    # GTA partial-noising: start from noise mixed with x1
    B = x1_t.shape[0]
    t_start = 1.0 - noise_ratio
    x = (1.0 - t_start) * torch.randn_like(x1_t) + t_start * x1_t

    # Sub-steps over [t_start, 1.0]
    sub_nfe = max(1, int(round(nfe * noise_ratio)))
    dt = (1.0 - t_start) / sub_nfe

    for i in range(sub_nfe):
        t_curr_scalar = t_start + i * dt
        t_next_scalar = min(t_curr_scalar + dt, 1.0)
        t_curr = torch.full((B,), t_curr_scalar, device=device)
        t_next = torch.full((B,), t_next_scalar, device=device)

        with torch.no_grad():
            v_uncond = cfm._velocity(x, t_curr, cond, cfg_scale)

        # Ramp guidance UP with t (only late steps have meaningful Q-grad).
        if guidance_schedule in ("linear-ramp", "linear-decay"):
            lam = guidance_scale * t_curr_scalar
        else:
            lam = guidance_scale

        if lam > 0:
            x_g = x.detach().requires_grad_(True)
            with torch.enable_grad():
                obs_flat = x_g[..., :obs_dim].reshape(-1, obs_dim)
                act_flat = x_g[..., obs_dim:obs_dim + action_dim].reshape(-1, action_dim)
                q_vals = q_fn(obs_flat, act_flat)              # (B*T,)
                q_vals.sum().backward()
            grad = x_g.grad.detach()                            # (B, T, D)

            # Apply to both obs and action sub-vectors. Normalize each
            # sub-vector per (b, t) for scale stability — same trick as V-guidance.
            guidance = torch.zeros_like(v_uncond)
            for slc in (slice(0, obs_dim), slice(obs_dim, obs_dim + action_dim)):
                g_slc = grad[..., slc]
                g_norm = g_slc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                guidance[..., slc] = lam * (g_slc / g_norm)
            v_uncond = v_uncond + guidance

        with torch.no_grad():
            x_pred = x + v_uncond * dt
            v_next = cfm._velocity(x_pred, t_next, cond, cfg_scale)
            x      = x + 0.5 * (v_uncond + v_next) * dt

    return x


def td_relabel_rewards(
    observations:      np.ndarray,   # (N, obs_dim)
    next_observations: np.ndarray,   # (N, obs_dim)
    value_fn:          Callable,     # takes torch.Tensor (B, obs_dim) → (B,)
    device:            torch.device,
    discount:          float = 0.99,
    batch_size:        int   = 4096,
) -> np.ndarray:
    """
    r_TD = V(s) - gamma * V(s')

    Why this works: V_phi is in-distribution by construction (it was
    trained on real data). Whatever reward the diffusion / reward_computer
    would have assigned, replacing it with the critic-induced one keeps
    the Q-update consistent on synthetic transitions.

    NOTE: this is the *negative* TD residual convention because we want
    r + γ V(s') ≈ V(s) on stationary policies, so r ≈ V(s) - γ V(s').
    """
    n = len(observations)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s   = torch.from_numpy(observations[start:end]).float().to(device)
            sp  = torch.from_numpy(next_observations[start:end]).float().to(device)
            v_s  = value_fn(s).cpu().numpy()
            v_sp = value_fn(sp).cpu().numpy()
            out[start:end] = v_s - discount * v_sp
    return out


def q_anomaly_mask(
    observations: np.ndarray,
    actions:      np.ndarray,
    q_fn:         Callable,        # (s, a) -> q value
    v_fn:         Callable,        # (s) -> value
    device:       torch.device,
    real_obs:     np.ndarray,
    real_act:     np.ndarray,
    quantile:     float = 0.99,
    batch_size:   int   = 4096,
) -> np.ndarray:
    """
    Build a boolean mask of transitions to KEEP.

    A transition is kept iff |Q(s,a) - V(s)| ≤ quantile-th percentile of
    the same quantity on the real dataset. This catches synthetic
    transitions where the critic is "surprised" (either over-confident
    OOD samples or under-confident absurd actions).
    """
    def batched(fn1, fn2, X1, X2=None):
        n = len(X1); out = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                t1 = torch.from_numpy(X1[s:e]).float().to(device)
                if X2 is not None:
                    t2 = torch.from_numpy(X2[s:e]).float().to(device)
                    out[s:e] = (fn1(t1, t2) - fn2(t1)).cpu().numpy()
                else:
                    out[s:e] = fn1(t1).cpu().numpy()
        return out

    real_resid = batched(q_fn, v_fn, real_obs, real_act)
    thr = float(np.quantile(np.abs(real_resid), quantile))

    syn_resid = batched(q_fn, v_fn, observations, actions)
    keep = np.abs(syn_resid) <= thr
    print(f"  Q-anomaly filter: kept {keep.sum():,}/{len(keep):,} "
          f"({100*keep.mean():.1f}%)  threshold |Q-V|≤{thr:.3f}")
    return keep

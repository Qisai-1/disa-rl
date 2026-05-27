# DiSA-RL — Method v2 Proposal: Critic-Anchored Proposal Augmentation (CAPA)

> Status: **proposal, awaiting sign-off.** No method code changed yet.
> Drafted 2026-05-21 while the v3 + Tier-1 sweep runs. Working name **CAPA**
> is a placeholder. See `project-progress-log`, `drc-method-fix`,
> `project-perf-improvements` for the chain of reasoning behind this pivot.

---

## 1. Why pivot away from DRC-IQL

DRC-IQL (the current v3 method) underperforms the real-only baseline on every
env, stable across the 2026-05-12 and 2026-05-18 runs. The cause is structural,
not a bug — verified against the code in `iql/agent.py`:

- **The critic is contaminated by synthetic data.** `update()` trains V on the
  *mixed* real+syn batch (`agent.py:207-226`), then bootstraps it into the
  Q-target through `V(next_obs)` (`agent.py:232`). Synthetic transitions —
  optimistically biased and variance-compressed — therefore pessimize/distort
  the *entire* value function, including its predictions on real states.
- **Both DRC mechanisms are purely defensive.** The density-ratio TD weight
  (`agent.py:236-240`) down-weights exactly the syn transitions that are rare
  under real data — i.e. the new coverage the diffusion model was built to add.
  The mixture expectile (`expectile_syn 0.5`) just leaks conservatism into real
  Q-targets via the shared V. Neither mechanism can ever make the augmented run
  *exceed* the baseline; the best case is "match it, with noise."

The Tier-1 ablation now running (`--alpha 0.15`, no `--sa_iql`) tests this
diagnosis. If Tier-1 ≥ baseline, the diagnosis is confirmed and CAPA below is
the fix. If Tier-1 still loses, the synthetic *data* is the problem and we go to
coverage-targeted generation first (out of scope here).

## 2. The core idea

**Keep synthetic data entirely out of the critic; let it influence only the
policy, judged by the real-only critic.**

A synthetic transition `(s, a, s')` does two separable things: it carries a
*reward/dynamics label* (untrustworthy — generated) and it carries a
*state-action proposal* `(s, a)` (potentially valuable — a place the policy
could act). DRC-IQL consumes both and defends against the first. CAPA consumes
only the second.

CAPA = **(i)** a critic trained on real data only — bit-for-bit the baseline
IQL critic — plus **(ii)** an actor whose AWR objective ranges over the mixed
real+syn batch, with every advantage scored by that trusted real critic, and
**(iii)** an uncertainty gate that discounts syn proposals the critic ensemble
disagrees about.

## 3. Algorithm — per-step `update()`

Inputs each step: `mixed_batch` (real+syn, fraction set by `alpha`) and a
full-size `real_batch` (already sampled today for the BC anchor via
`aug_buffer.sample_real`).

1. **V update — real only.** Expectile regression of `Q_target.min(s,a) − V(s)`
   on `real_batch`. Standard IQL expectile (0.7). No mixture expectile.
2. **Q update — real only.** TD target `r + γ(1−d)·V(s')` on `real_batch`.
   Because V is now real-only, the bootstrap is clean. Keep the PA loss
   (`pa_weight`) — it is a real-data support constraint, unrelated to syn data.
3. **Actor update — AWR over the mixed batch.** For every `(s,a)` in
   `mixed_batch` (real *and* syn), advantage `A(s,a) = Q_target.min(s,a) − V(s)`
   from the real critic; AWR weight `exp(temperature·A)`.
   - **Uncertainty gate (new).** With the 10-critic QEnsemble already in use,
     compute `σ_Q(s,a) = std` over critics. Multiply the AWR weight of *syn*
     rows by `g(σ_Q) = exp(−β·σ_Q)` (or hard-drop above a percentile). Real
     rows are ungated. High ensemble disagreement = epistemic OOD = the critic
     cannot be trusted to rank that proposal, so it is discounted.
4. **BC anchor — real only.** Unchanged (`agent.py:320-332`).

## 4. The guarantee — why CAPA cannot underperform the baseline

The CAPA critic sees *only* real data, the same network architecture, the same
expectile, and (with a full-size `real_batch`) the same per-step sample budget
as the `offline_only` baseline. It is therefore the **same estimator** as the
baseline critic — identical in distribution. Synthetic data enters *only* the
actor's AWR loss as additional `(s,a)` pairs to imitate-weight.

Consequences:
- The augmented run's critic provably matches the baseline critic. Any score
  delta is attributable solely to the actor seeing more proposals.
- **Synthetic rewards are never consumed by any loss.** AWR uses `Q−V`, not
  reward; the Q-target uses only real rewards. The optimistic-reward bias of the
  syn data — the single most damaging syn-data pathology we measured — is
  *structurally* removed, not corrected-for.
- Worst case for syn data: the proposals are useless and the uncertainty gate
  zeroes them → CAPA degenerates to the baseline. It cannot do worse.

This "monotone improvement over the base offline critic" is the paper's
headline claim and is defensible because it is structural, not empirical.

## 5. Exact code changes (for sign-off — not yet applied)

Small and localized. Two files:

**`iql/agent.py` — `update()`**
- New signature: `update(mixed_batch, real_batch, *, capa: bool)`.
- When `capa=True`: run steps 1–2 (V, Q) on `real_batch` instead of `batch`;
  run step 3 (AWR) on `mixed_batch`; skip the SA-IQL discriminator entirely.
- Uncertainty gate: add a helper `q_ensemble_std(obs, action)` on the agent
  (the QEnsemble already exposes per-critic Q via `self.q.all` → just `.std(0)`),
  and apply `g(σ_Q)` to syn rows of the AWR `weight` (use `batch["source"]`).
- New ctor args: `capa: bool = False`, `unc_beta: float` (gate strength).

**`iql/train_iql.py`**
- New flag `--capa` (and `--unc_beta`). In CAPA mode, always sample a full
  `real_batch` (not gated on `bc_weight`) and pass both batches to `update()`.
- `--capa` is mutually exclusive with `--sa_iql`; assert.

**No change to** the diffusion model, `generate_synthetic_data.py`, the
discriminator, or the checkpoint format. `arch` dict in `save()` should gain a
`capa` flag for clean reload, but old checkpoints stay loadable.

## 6. Experiment plan

Once Tier-1 confirms the diagnosis:
- **CAPA sweep**: 8 envs (medium-v2 + medium-replay-v2), seed 0, `--capa`,
  `--alpha 0.5`, uncertainty gate on. Same critic arch/hyperparams as baseline.
- **Ablations** (all vs the same baseline): CAPA without the uncertainty gate;
  CAPA with critic on the *mixed* batch (isolates the real-only-critic effect);
  α ∈ {0.25, 0.5, 0.75} (proposal mixing fraction — note α now only affects the
  actor, so larger α is safe).
- **Comparators**: real-only baseline, v3 DRC-IQL, PARS, SynthER-style naive
  augmentation (no critic anchoring). 3-way table → 5-way table.
- Multi-seed (3–5) only for the final CAPA + baseline + PARS cells.

## 7. Risks / open questions

- **Syn-state advantage extrapolation.** `Q(s,a)` and `V(s)` for syn `s` are
  off-support; the uncertainty gate mitigates this but does not eliminate it.
  Fallback: also gate on `V(s)` ensemble disagreement, or only accept syn `s`
  within the real obs per-dim range (we have those stats).
- **Does CAPA actually *beat* the baseline, not just match it?** The guarantee
  is only ≥. Improvement requires the syn proposals to surface genuinely better
  actions in real-reachable states. If they do not, we need coverage-targeted
  generation (condition the diffusion model on the critic's low-density /
  high-advantage region) — the natural follow-up contribution.
- **Naming.** CAPA is a placeholder. Decide before the writeup.

## 8. What CAPA deliberately drops

The density-ratio discriminator, the mixture expectile, and `--sa_iql` as a
whole. They were the v3 novelty but they are net-negative. CAPA's novelty is the
*monotone-improvement guarantee* + *uncertainty-gated proposal acceptance* — a
cleaner and more defensible contribution for the AAAI submission.

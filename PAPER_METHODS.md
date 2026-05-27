# DiSA-RL — Methods Section (draft)

> Working title: **Density-Calibrated Diffusion Augmentation for Offline-to-Online RL**

---

## 1. Problem Setup

We consider the standard offline-to-online RL setting. An agent is given a
static dataset
`D_real = {(s_i, a_i, r_i, s_i')}_{i=1}^N`
collected by an unknown behaviour policy. After offline training, the agent
is allowed `B_online` real environment interactions for fine-tuning. The
goal is to maximise the expected discounted return of the final policy.

We augment `D_real` with a synthetic dataset `D_syn` produced by a
trajectory diffusion model `g_θ`. The full training set is
`D = D_real ∪ D_syn`. The challenge: `D_syn` is OOD by construction
(generated, not collected), and the offline RL critic is OOD-blind.

---

## 2. DiSA-RL Framework Overview

DiSA-RL closes the loop **critic → generator → critic** through three
coupled mechanisms:

> ![framework]
>
> 1. **Density-Ratio-Calibrated IQL (DRC-IQL).** A learned discriminator
>    `d_ψ(s,a)` provides per-transition importance weights `w(s,a)` that
>    correct the Bellman target on synthetic data. A source-aware
>    mixture expectile gives optimistic V-estimates on real and
>    conservative ones on synthetic, anchoring the critic to the
>    in-support manifold.
> 2. **Q-Conditional Diffusion (QCD).** The trajectory diffusion model
>    is conditioned on a learned Q-value (`Q_φ(s_0, a_0)`) rather than
>    on raw return. The conditioning signal is shaped by the same
>    critic that will consume the generated data, closing one half of
>    the loop.
> 3. **Trust-Region Iterative Refinement (TR-IVAR).** The DRC-IQL → QCD
>    cycle iterates: each round trains a fresh critic on the previous
>    round's augmented data and uses it to generate the next round's
>    syn data. A per-round trust-region constraint on the density
>    ratio bounds the drift and yields monotone-improvement guarantees
>    under realizability.

The three pieces are independently ablatable; each removes one source of
distribution shift in the offline RL augmentation loop.

---

## 3. Pillar 1 — DRC-IQL

### 3.1 Background: Implicit Q-Learning (IQL)

IQL (Kostrikov et al., 2022) maintains a state-value `V_φ(s)`, twin Q
`Q_θ(s,a)`, and Gaussian actor `π_ω(a|s)`. Updates:

- **V (expectile regression):**
  `L_V(φ) = E_{(s,a)~D} [ L^τ(Q̄(s,a) − V_φ(s)) ]`
- **Q (TD with V target):**
  `L_Q(θ) = E_{(s,a,r,s')~D} [ (r + γ·V_φ(s') − Q_θ(s,a))² ]`
- **Actor (AWR):**
  `L_π(ω) = E_{(s,a)~D} [ −exp(β·A(s,a)) · log π_ω(a|s) ]`
  where `A(s,a) = Q̄(s,a) − V_φ(s)`.

### 3.2 DRC-IQL — what we change

**(a) Density-ratio importance weighting on Q.** We train a small
classifier `d_ψ : (s,a) → [0,1]` on the mixed batch with BCE loss
(label 1 = real, 0 = syn). The implied density ratio is

`w(s,a) = clip( d_ψ(s,a) / (1 − d_ψ(s,a)) , [w_min, w_max] )`

and the Q-loss becomes a source-weighted IS-corrected version:

`L_Q(θ) = E_real [(r + γV(s') − Q)²]  +  E_syn [ w · (r + γV(s') − Q)² ]`.

Under realizability of `w`, this is an unbiased estimator of the
real-data Bellman error (Lemma 1).

**(b) Mixture expectile on V.** We use a per-source expectile:

`τ(source) = τ_real · 1[source=real] + τ_syn · 1[source=syn]`

with `τ_real = 0.9` (optimistic on data we trust) and
`τ_syn = 0.5` (median, neither optimistic nor pessimistic on syn).

### 3.3 Lemma 1 (informal)

Under realizability of `w(s,a) = p_real(s,a) / p_syn(s,a)`,
`E_syn[ w · L(s,a) ] = E_real[ L(s,a) ]` for any bounded loss `L`.

*Proof sketch.* Change-of-measure identity for absolutely continuous
distributions. The clip `[w_min, w_max]` introduces bias bounded by
the truncation tails — controlled by Bernstein's inequality on
`|w − w_clip|`.

---

## 4. Pillar 2 — QCD (Q-Conditional Trajectory Diffusion)

### 4.1 Background: trajectory diffusion

A trajectory diffusion model `g_θ : (z, c) → x` learns to denoise a
trajectory `x ∈ R^{T × (|S|+|A|)}` from noise `z`, conditioned on
`c = [s_0_norm, scalar_norm]`. The scalar is traditionally:

- **Return** (Decision Diffuser, SynthER, GTA, GODA): `R = Σ_t r_t`,
  z-scored by training-set statistics.

### 4.2 QCD — what we change

We replace the return scalar with a **per-trajectory Q-target** drawn
from the offline-trained DRC-IQL critic:

`c = [s_0_norm,   (Q_φ(s_0, a_0) − μ_Q) / σ_Q ]`

where `(μ_Q, σ_Q)` are the empirical Q-distribution moments on the
training sub-trajectories. At inference, we draw `Q*` from the top-20 %
of the empirical Q-distribution and run conditional diffusion under
`c = [s_0, (Q* − μ_Q)/σ_Q]`.

### 4.3 Why this beats return-conditioning

- Returns are dataset-bounded; Q-targets are critic-bounded. The
  critic has already absorbed the pessimism penalty on OOD actions,
  so high Q corresponds to "in-support, high-value" by construction.
- Returns are trajectory-level; Q at `s_0` reflects the entire future
  rollout under the critic. The conditioning thus carries more
  information per sub-trajectory.
- The same critic that ranks the syn data downstream (DRC-IQL) is the
  one that conditions the generator. The loop is closed.

### 4.4 Architecture

QCD requires no model-architecture change to the trajectory diffusion
model. Only the conditioning vector's *scalar* changes; the dim stays
at `|S| + 1`. This lets QCD be a drop-in replacement on top of any
existing return-conditioned trajectory diffuser.

---

## 5. Pillar 3 — TR-IVAR (Trust-Region Iterative Refinement)

### 5.1 The outer loop

```
Round 0:   train V_φ⁰, Q_φ⁰ on D_real (real-only DRC-IQL).
Round k≥1:
   1. Generate D_syn^k ← g_θ ( · | Q_φ^{k-1})
   2. Train DRC-IQL on D_real ∪ D_syn^k        → V_φ^k, Q_φ^k
   3. Estimate ε^k = E_{(s,a)~D_syn^k} [(w^k − 1)²]
   4. If ε^k > ε_max:  attenuate λ_v *= ½, α *= 0.75, regenerate.
                        else: accept and advance.
```

### 5.2 Trust region

We bound the density-ratio drift per round. The intuition: if the
generator chases a moving critic too aggressively, the new syn
distribution may have negligible mass under the previous round's
realizable function class, voiding the unbiased-estimation property of
Lemma 1. The trust region ensures
`KL(p_syn^k ‖ p_syn^{k-1}) ≤ ε_max`.

### 5.3 Theorem 2 (informal — monotone improvement)

Let `π^k = π_ω^k` be the policy at the end of round `k`. Under
realizability of the density ratio and Lipschitz continuity of the
Q-network in parameter space, if `ε^k ≤ ε`, then

`J(π^k) ≥ J(π^{k-1}) − c · √ε`

for a problem-dependent constant `c` (Schulman et al. 2015–style
performance-difference bound).

---

## 6. Offline-to-Online Phase

After `K` rounds offline, we initialise a SAC actor-critic from the
final DRC-IQL checkpoint. The first two critics of the Q-ensemble seed
SAC's twin Q (`create_sac_from_iql` in our code). SAC then fine-tunes
on real env data while:

1. A background process continues running QCD generation conditioned
   on the current online critic `Q^t`.
2. The diffusion model is fine-tuned with EWC on the freshly collected
   real rollouts (existing `online_rl/async_generator.py`).
3. The adaptive mixing ratio `ρ` (fraction of synthetic samples in
   each batch) follows the schedule of Janner et al. (2019),
   `ρ = η(1 − γ) / ε_roll`, where `ε_roll` is estimated as MMD between
   real and syn observations.

This phase is the same closed loop as Pillar 3, just running with
SAC instead of IQL after the offline-to-online transition.

---

## 7. Practical Building Blocks (adopted from prior work)

Engineering pieces we use **as building blocks** (not claimed as
contributions):

- **Wider critic with LayerNorm** (folklore; +2-4 pts D4RL).
  Q ensemble of `M = 10` critics with 4×256 hidden, LayerNorm after
  each linear layer (REDQ; Chen et al. 2021).
- **Reward scaling** (`c_reward = 5 - 10` depending on env; PARS,
  ICML 2025). Multiplies reward magnitudes, sharpens Q-value
  calibration of OOD actions through LN.
- **PA loss** (PARS, ICML 2025). Penalises Q-values at infeasible
  actions (uniform in `[-2, -1] ∪ [+1, +2]`) toward a constant lower
  bound `Q_min`.
- **Advantage normalization** in AWR (folklore; stabilizes early
  training).
- **Action-noise augmentation** (folklore; small Gaussian noise on
  dataset actions in the AWR log-prob acts as a Jacobian
  regularizer).
- **GPU-resident replay buffer** (engineering; 3-5x training speedup
  by eliminating host↔device transfers).
- **Alpha warmup schedule** for the synthetic mixing ratio (our
  engineering; eliminates the augmented-vs-baseline early-training
  slowdown).

These are independently useful but well-known. Our novelty rests on
the three pillars in §3, §4, §5.

---

## 8. Experimental Setup

### 8.1 Environments

D4RL MuJoCo locomotion: `halfcheetah / hopper / walker2d / ant`,
each at `medium-v2` and `medium-replay-v2` (8 settings).

### 8.2 Comparisons

| Group | Method | Notes |
|---|---|---|
| Offline RL baselines | BC, TD3+BC, IQL, CQL | Numbers from Kostrikov et al. (2022). |
| Strong offline RL | PARS (LG AI, ICML 2025) | Re-run with their public code. |
| Diffusion augmentation | SynthER (NeurIPS 2023) | Re-run with public code. |
|  | GTA (NeurIPS 2024) | Re-run if open source. |
| Offline-to-online | Cal-QL, RLPD | Numbers from respective papers. |
| **Ours** | DRC-IQL, +QCD, +TR-IVAR, +online | Each pillar reported separately for ablation. |

### 8.3 Protocol

- 5 random seeds per cell.
- 500 k offline training steps; 100 k online fine-tuning steps for the
  offline-to-online column.
- Normalised score per D4RL convention
  (`(R − R_random) / (R_expert − R_random) · 100`).
- Eval every 10 k steps × 10 episodes; report mean ± std at the best
  checkpoint per seed and final-step score.

---

## 9. Open questions for the writing pass

1. **Theorem 2 statement.** Tighten the assumption on Lipschitz Q in
   parameter space. Cite Schulman 2015 + a more modern bound.
2. **Sample complexity of `d_ψ`.** Bound the discriminator's
   generalisation gap as a function of dataset size, and propagate
   to the IS-correction error. Likely a few hundred to a few
   thousand iterations is enough in practice (we use 1 BCE step per
   training iteration); verify empirically.
3. **Failure modes.** What if `w` saturates at the clip boundary on
   ant (high-dim obs)? Plot `w_mean` vs training step for all envs to
   diagnose.
4. **Reward scaling × DRC interaction.** PARS reports gains via
   c_reward + LN. Does this compound with DRC-IQL, or are they
   substitutes (both regularising Q on OOD)? Important ablation.

---

## 10. Code map

| File | Pillar | Notes |
|---|---|---|
| `iql/agent.py`         | 1 | DRC-IQL agent: mixture expectile, density-ratio TD weight, Q-ensemble, PA loss. |
| `iql/discriminator.py` | 1 | Density-ratio discriminator. |
| `iql/networks.py`      | – | TwinQ + QEnsemble + GaussianActor with LayerNorm. |
| `iql/buffer.py`        | – | GPU-resident replay buffer + alpha warmup + reward scaling. |
| `iql/train_iql.py`     | – | Single entry-point with all flags. |
| `iql/ivar.py`          | 3 | TR-IVAR orchestrator. |
| `diffusion/q_conditional.py` | 2 | Q-target computation + sampling. |
| `diffusion/data.py`    | 2 | Normalizer Q-fields + TrajectoryDataset Q-cond. |
| `diffusion/train.py`   | 2 | `--qcd` flag + auto Q-target dataset. |
| `diffusion/generate.py`| 2 | Auto-dispatch on `cond_kind` at inference. |
| `online_rl/sac.py`     | online | SAC + `create_sac_from_iql` factory. |
| `online_rl/train_online.py`, `async_generator.py`, `online_buffer.py` | online | offline→online + EWC fine-tuning. |
| `generate_synthetic_data.py` | – | Unified VCDG/QCD generation entry-point. |

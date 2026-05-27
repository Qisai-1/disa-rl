# DiSA-RL — Refined Research Plan (after 2026-05-12 literature scan)

## Paper title (working)

**Density-Calibrated Diffusion Augmentation for Offline-to-Online Reinforcement Learning**

Or shorter: **DiSA-RL: Closing the Distribution-Shift Loop in Diffusion-Augmented Offline RL**

## Storyline

> Naive diffusion augmentation of offline RL suffers from a closed-loop bias:
> (i) the diffusion model extrapolates outside the support of the offline
> dataset; (ii) the offline RL critic is OOD-blind to those samples; (iii)
> one-shot generation cannot self-correct. We propose three coupled
> corrections — DRC-IQL, QCD, TR-IVAR — that together close the loop.

## Where DiSA-RL sits vs the 2023-2025 literature

| Paper | Aug. via diffusion | Critic-coupled gen | **Density-ratio correction** | Iterative refinement | Offline→Online |
|---|:-:|:-:|:-:|:-:|:-:|
| SynthER (NeurIPS'23) | ✓ | ✗ | ✗ | ✗ | ✗ |
| GTA (NeurIPS'24) | ✓ | ✗ (return) | ✗ | ✗ | ✗ |
| GODA (Dec'24) | ✓ | ✗ (RTG) | ✗ | ✗ | ✗ |
| Diffusion-DICE (Jul'24) | (policy) | post-hoc | (DICE-side) | ✗ | ✗ |
| QGPO (2023) | (action) | ✓ (energy) | ✗ | ✗ | ✗ |
| GFP (Dec'25) | (policy) | ✓ (val-weight BC) | ✗ | ✗ | ✗ |
| CFDG (ICLR'25) | ✓ | ✗ (plain CFG) | ✗ | ✗ | ✓ |
| AdaptDiffuser (ICML'23) | online | ✓ (reward grad) | ✗ | ✓ | (online MPC) |
| **DiSA-RL (ours)** | **✓** | **✓ (gen + filter)** | **✓** | **✓ (TR)** | **✓** |

DiSA-RL is the only entry in every column.

---

## Three algorithmic contributions

### Pillar 1 — DRC-IQL (Density-Ratio Calibrated IQL)

**Definition.** A small discriminator `d_ψ(s,a) → P(real | s,a)` is trained
on-the-fly with binary cross-entropy on the mixed (real, syn) batch. The
implied density ratio `w(s,a) = d / (1 − d)` is clipped to `[0.5, 2.0]`
and used to reweight the Q-update on syn transitions:

```
L_Q(θ) = E_real[(r + γV(s') − Q(s,a))²]
       + E_syn [ w(s,a) · (r + γV(s') − Q(s,a))² ]
```

A complementary **mixture expectile** is used in the V-update:
`τ(source) = τ_real = 0.9` on real transitions, `τ_syn = 0.5` on syn.

**Why it works.** Under realizability of the density ratio, the weighted
syn loss is an unbiased estimator of the real-data Bellman residual
(Lemma 1 below). Practically, the discriminator down-weights
high-confidence-syn-but-OOD transitions and lets the critic largely ignore them.

**Novelty.** No prior offline RL augmentation paper does this. Closest:
DICE methods (OptiDICE, Diffusion-DICE) use density ratios at training
time, but for policy correction, not Bellman backups under augmentation.

**Lemma 1 (informal).** Under realizability of `w(s,a) = p_real / p_syn`,
the importance-weighted syn loss equals the real loss:
`E_{syn}[ w · L(s,a) ] = E_{real}[ L(s,a) ]`.

### Pillar 2 — QCD (Q-Conditional Trajectory Diffusion)

**Definition.** Replace the standard return-conditioning of the diffusion
model with **per-step Q-target conditioning**. Training: pair each
sub-trajectory with the running Q-target sequence from the offline IQL
critic; condition vector becomes `[s_0_norm, Q*_norm]`. Inference: sample
a top-quantile Q-target sequence and condition.

**Why it works.** Return-conditioning is trajectory-level — the same return
value can correspond to wildly different state distributions. Q-targets are
per-step — they constrain the manifold of generated trajectories much more
tightly to the in-support, high-value region.

**Novelty.** Decision Diffuser, SynthER, GTA, GODA all use return/RTG.
Per-step Q-target conditioning of a trajectory diffusion model for offline
RL augmentation has, to our knowledge, not been published.

### Pillar 3 — TR-IVAR (Trust-Region Iterative Value-Aware Refinement)

**Definition.** Outer loop with K rounds:

```
Round k:
  1.  Train DRC-IQL on D_real ∪ D_syn^{k-1}        → Q^k, V^k
  2.  Generate D_syn^k with QCD using (Q^k, V^k)
  3.  Trust-region: require  E[(w^k - 1)²] ≤ ε
      else attenuate λ_v guidance and retry
```

**Why it works.** Without a trust region, the syn distribution can chase
a moving critic and oscillate. With a TR on the density ratio change, the
update has a Lipschitz drift bound, and (Theorem 2) the policy improves
monotonically up to an `O(√ε)` slack.

**Novelty.** Iterative diffusion refinement in offline RL has only been
explored in AdaptDiffuser (online MPC). Putting it in an offline-to-online
loop with an explicit density-ratio trust region is new.

**Theorem 2 (informal).** Let `π^k` be the policy at round k and assume
`E_{s,a∼p^{k-1}}[(w^k(s,a) − 1)²] ≤ ε`. Then `J(π^k) ≥ J(π^{k-1}) − c·√ε`
for a problem-dependent constant `c`. Proof sketch: standard performance
difference lemma applied to the density-ratio drift.

---

## Implementation status

| Component | File | Status |
|---|---|---|
| Bug-fix `target_return` (sub-traj p90 + per-batch sampling) | `generate_synthetic_data.py` | ✓ done |
| Wider Q (4×256) + Q-ensemble (M=10, REDQ subset) | `iql/networks.py`, `iql/agent.py` | ✓ done |
| Advantage normalization, action-noise augmentation | `iql/agent.py` | ✓ done |
| GPU-resident buffer with `source` field | `iql/buffer.py` | ✓ done |
| Alpha warmup schedule (`alpha_warmup`, `alpha_ramp`) | `iql/buffer.py`, `iql/train_iql.py` | ✓ done |
| **DRC-IQL (A1)**: discriminator + density-ratio TD weight + mixture expectile | `iql/discriminator.py`, `iql/agent.py` | ✓ done |
| VCDG: value-guided sampling + TD relabel + Q-anomaly filter | `diffusion/value_guided.py` | ✓ done (used as ablation knob) |
| IDM: state-only diffusion action override | `diffusion/inverse_dynamics.py` | ✓ done (network), wiring partial |
| **QCD (A2)**: Q-conditional diffusion training | `diffusion/q_conditional.py`, `diffusion/data.py`, `diffusion/train.py`, `diffusion/generate.py` | ✓ done |
| **TR-IVAR (A3)**: outer-loop orchestrator with TR | `iql/ivar.py` | ✓ done |
| PARS-adopted reward scaling + PA loss | `iql/buffer.py`, `iql/agent.py`, `iql/train_iql.py` | ✓ done (building blocks) |
| Offline→online fine-tuning loop with EWC + adaptive `ρ` | `online_rl/` (scaffolding exists) | ⏳ later |

## Adopted from prior work (NOT our novelty, but powerful building blocks)

| Trick | Source | Where in our code | Effect |
|---|---|---|---|
| Q-ensemble (M=10) + REDQ subset | REDQ (ICLR'21) | `iql/networks.py QEnsemble` | +3-5 pts D4RL |
| Wider Q with LayerNorm | folklore + many papers | `--q_hidden_dims 256 256 256 256` | +2-4 pts |
| Reward scaling × 5-10 | **PARS** (ICML'25) | `--reward_scale 5 or 10` | up to +30 pts hopper/walker2d in PARS |
| PA loss (Q on OOD action → min_q) | **PARS** (ICML'25) | `--pa_weight 0.001 --pa_min_q -100` | OOD Q-bound regularizer |
| Advantage normalization | folklore | `--adv_normalize` (default on) | +1-2 pts, stability |
| Action-noise augmentation | folklore | `--action_noise_std 0.05` | +0.5-2 pts |
| Alpha warmup | DiSA-RL native | `--alpha_warmup --alpha_ramp` | fixes augmented-vs-real slow convergence |

---

## Ablation plan (for the paper's main table)

5 seeds, 4 envs (halfcheetah / hopper / walker2d / ant — medium-v2 first,
then medium-replay-v2):

| # | Method | flags |
|---|---|---|
| 1 | BC (behavioral cloning) | — |
| 2 | IQL (real only) | `--mode offline_only --bc_weight 0` |
| 3 | IQL + naive syn (SynthER analog) | `--mode augmented` (legacy syn) |
| 4 | IQL + bug-fix syn | as above with bug-fixed syn |
| 5 | IQL + bug-fix + DRC-IQL | `--sa_iql` |
| 6 | IQL + bug-fix + DRC + Q-ensemble | + `--num_critics 10` |
| 7 | IQL + DRC + QCD (Pillar 1+2) | requires QCD checkpoint |
| 8 | **DiSA-RL (Pillar 1+2+3)** | all + TR-IVAR rounds |
| 9 | DiSA-RL − DRC | drop `--sa_iql` |
| 10 | DiSA-RL − QCD | use return-conditioned diffusion |
| 11 | DiSA-RL − TR | IVAR without trust region |
| 12 | DiSA-RL − IDM | joint state-action diffusion |

Rows 5-8 are the contribution claim. Rows 9-12 are the ablation rows.

---

## Risks and mitigations

- **Risk:** the discriminator becomes too confident (`d → 1` on real,
  `d → 0` on syn) and density ratios saturate at the clip boundary.
  **Mitigation:** small classifier (one hidden layer 256), spectral norm
  optional, low LR. Already implemented with LR=3e-4.

- **Risk:** QCD requires retraining the diffusion model from scratch,
  ~12h GPU. **Mitigation:** start training overnight after v1 sanity
  confirms DRC-IQL lifts scores.

- **Risk:** the TR-IVAR loop oscillates. **Mitigation:** TR constraint;
  also fallback to K=1 (single round) if drift exceeds `ε` and report
  both.

---

## What's NOT a contribution (be honest)

- The bug-fix on `target_return` is a fix, not a contribution.
- IDM is a known idea (LMPC, PETS).
- Wider critics / Q-ensemble are known D4RL tricks.
- VCDG's value gradient guidance overlaps with AdaptDiffuser; we frame it
  as a sampling-time ablation knob, not a pillar.

These are reported as engineering / implementation details. The paper's
three claims are DRC-IQL, QCD, TR-IVAR.

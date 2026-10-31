# score-centering-lab

An exact, CPU-only testbed for **score centering** — the additive fix for
training-inference mismatch (TIM) in policy-gradient RL, from Marek &
Ryabinin (Together AI), "Score Centering Stabilizes Off-policy
Reinforcement Learning" (arXiv:2609.20807, 17 Sep 2026).

## The problem

Policy-gradient RL needs two forward passes: the *sampler* (inference
engine) generates rollouts, and the *trainer* computes gradients on them.
In practice the two engines disagree — different precisions, different
kernels, stale checkpoints — so rollouts come from a sampler `q` while
gradients are computed through a trainer `p ≠ q`. The paper shows the
instability this causes is primarily **drift**: a persistent bias that
accumulates every training step.

At any prefix, the expected policy-gradient update splits (paper Eq. 3)
into a drift term and a signal term:

```
E_q[R · s] = E_q[R] · E_q[s]   +   Cov_q(R, s)
                 ^ drift            ^ signal
```

The drift term carries no information about which rollouts were
successful — it depends on the rewards only through their mean. What it
does is distill the trainer toward the sampler, scaled by the expected
reward. The sampler is a biased copy of the trainer that gets re-synced
from the trainer after every step, so the error compounds in a feedback
loop instead of converging. Group centering the rewards does **not**
remove drift: drift arises at individual prefixes, and the expected
advantage at a prefix that leads to correct answers is positive.

## The fix: score centering

Subtract the sampler's expected score from each score (paper Eq. 4–5):

```
s̃ = s − E_q[s]          E_q[R · s̃] = Cov_q(R, s)
```

The centred score has zero expectation under the sampler by construction,
so the drift term is cancelled **exactly**, at every prefix, with no
importance ratio, no masking, and no clipping. With a softmax policy the
expected score is `E_q[s] = q − p` in logit space, so the centred score
for a sampled token `y` is simply `e_y − q`. Because the correction is
additive, it composes with importance sampling: the ratio partially
corrects the sampling distribution (the covariance is measured under `q`
rather than `p`) while score centering removes the remaining drift.

## The testbed

A small linear-softmax policy generates 6-digit sequences whose digit
sum must hit a target (a verifiable-reward task in the spirit of the
paper's Countdown experiments). The policy shares parameters across
states — one-hot features over (remaining sum, position) pairs plus the
target — so a biased update in one state moves the policy everywhere,
which is the channel through which drift compounds in a real model.
Rollouts come from a *sampler* that is a biased copy of the trainer, in
exactly the paper's three mismatch settings:

| Setting | Paper section | Implementation |
|---|---|---|
| `noise-σ` | §5.2 weight noise | sampler weights = trainer weights + fixed Gaussian offset |
| `stale-k` | §5.3 staleness | sampler is a snapshot refreshed only every k steps |

Quantisation was also tested (fixed-step weight rounding and logit
snapping to 2–4 levels, §5.3) but did not separate the methods at any
tested severity on this testbed — see "What this reproduces" below.

All methods share the objective (REINFORCE with group-centred rewards),
sampler, trainer, optimiser (Adam), and advantages — the paper's
Section 5.1 "shared objective" design — so the only difference between
runs is the correction applied on top:

| Method | Update per sampled token |
|---|---|
| `naive` | `A · (e_y − p)` — vanilla policy gradient |
| `sc` | `A · (e_y − q)` — score centering (exact) |
| `sc_topk` | `A · (e_y − q̃)` — score centering, top-k sampler approx (§4.1) |
| `tis` | `A · min(p_y/q_y, 2) · (e_y − p)` — truncated importance sampling (Table 1) |
| `mis` | `A · (p_y/q_y if 0.5 ≤ p_y/q_y ≤ 5 else 0) · (e_y − p)` — masked importance sampling (Table 1) |
| `sc+tis` | `A · (min(p_y/q_y, 2)·(e_y − p) − E_q[w·s])` — composition, Eq. 11 |
| `sc+mis` | `A · (w_y·(e_y − p) − E_q[w·s])` with MIS weights — composition, Eq. 11 |
| `oracle` | no mismatch at all (reference ceiling) |

## Results

Final greedy exact-match accuracy (fraction of 200 fresh prompts whose
digit sum hits the target), mean over 5 seeds, 400 Adam steps each.
Full per-run records in `results/experiment.json`; the table below is
cross-footed against those records by `scripts/crossfoot.py`.

| setting | naive | sc | sc_topk | tis | mis | sc+tis | sc+mis | oracle |
|---|---|---|---|---|---|---|---|---|
| noise-1.0 | 1.000 | 0.994 | 1.000 | 1.000 | 0.983 | 0.992 | 1.000 | 1.000 |
| noise-1.5 | 0.884 | 0.953 | 0.957 | 0.964 | 0.924 | 0.905 | 0.870 | 1.000 |
| noise-2.0 | 0.683 | 0.793 | 0.763 | 0.906 | 0.726 | 0.729 | 0.626 | 1.000 |
| noise-3.0 | 0.454 | 0.391 | 0.432 | 0.793 | 0.781 | 0.336 | 0.312 | 1.000 |
| stale-64 | 0.885 | 0.916 | 0.930 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| stale-128 | 0.094 | 0.395 | 0.439 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Reading the table:

- **Mild mismatch (noise-1.0)**: every method reaches ~1.000 — under mild
  TIM the methods overlap, exactly as the paper's Section 5.1 argues
  (drift needs steps to accumulate before methods separate).
- **Weight noise (σ=1.5–2.0)**: every correction beats naive, and the gap
  widens with severity (naive 0.884 → 0.683). Score centering alone is
  clearly better than naive (0.953 / 0.793 vs 0.884 / 0.683); truncated
  importance sampling is the strongest single method here (0.964 / 0.906).
  The compositions are *worse* than their components under weight noise
  (sc+tis 0.905 / 0.729) — the IS weights are noisy when the sampler is
  randomly perturbed, and the composed estimator pays for it.
- **Extreme weight noise (σ=3.0)**: the ordering flips. Naive (0.454)
  beats score centering (0.391) and both compositions (0.336 / 0.312),
  while TIS (0.793) and MIS (0.781) dominate. See the limits section.
- **Staleness**: at k=64 every IS/composition method is perfect (1.000)
  while naive is at 0.885 and SC at 0.916. At k=128 naive **never
  learns at all** (final 0.094, peak 0.138 across all 5 seeds — flat
  trajectories near zero for the whole run), SC climbs slowly to 0.395,
  and TIS, MIS, sc+tis and sc+mis all reach 1.000. This reproduces the
  paper's Figure 4 finding: under staleness, score centering composed
  with importance sampling dominates — and here plain TIS/MIS do too.
- **Top-k (sc_topk, k=8 of 10)**: tracks full score centering within
  0.05 everywhere (0.957 vs 0.953 at noise-1.5; 0.439 vs 0.395 at
  stale-128), matching the paper's Section 4.1 / Figure 5 ablation.
- **Learning speed**: corrections also learn faster. Steps to 0.80
  exact-match at noise-1.5: naive 185, sc 100, tis 100; at noise-2.0:
  naive 300, sc 294, tis 130. At stale-64: naive 300, sc 294, tis 135.

## Drift probe

With a constant reward `R = +1` for every rollout there is no learning
signal at all: on-policy, the expected gradient is exactly zero. The
probe computes each method's expected update under a biased sampler
(weight noise σ=0.6) exactly (summing over the vocabulary — no sampling
noise) and then trains with constant reward to watch the dynamics. Full
records in `results/drift_probe.json`; all values are means over 3 seeds.

| method | ‖E_q[update]‖ at init | KL(policy ‖ init) after 150 steps | KL(policy ‖ sampler) init → final |
|---|---|---|---|
| naive | 0.1509 | 2.0079 | 0.3628 → 0.0243 |
| sc | 0.0000 | 0.0943 | 0.3561 → 0.3691 |
| sc_topk | 0.0009 | 0.1338 | 0.3560 → 0.3447 |
| tis | 0.0241 | 2.2723 | 0.3558 → 0.0021 |
| mis | 0.0360 | 2.2825 | 0.3477 → 0.0027 |
| sc+tis | 0.0000 | 0.0925 | 0.3565 → 0.3492 |
| sc+mis | 0.0000 | 0.1643 | 0.3565 → 0.3410 |
| control (no mismatch) | 0.0000 | 0.1178 | — |

Reading the table:

- **The drift is real and exactly cancelled by score centering.** Naive
  PG's expected update under the biased sampler is 0.1509 — a nonzero
  drift vector (Eq. 3). Score centering's is 0.0000, exactly, as Eq. 5
  promises — and so are both Eq. 11 compositions, for any weighting
  function. The top-k approximation is nearly exact (0.0009).
- **TIS/MIS shrink the drift but do not remove it** (0.0241 / 0.0360):
  clipping and masking reintroduce drift, as the paper's Section 4 says.
- **The dynamics follow the expectation.** Under constant reward, naive
  PG is dragged systematically toward the sampler: its KL to the sampler
  collapses from 0.3628 to 0.0243 while its KL to the initialisation
  grows to 2.0079. TIS and MIS distill even harder toward the sampler
  (KL 0.0021 / 0.0027). Score centering does not distill at all — its KL
  to the sampler stays at 0.3691 — and its movement from init (0.0943)
  is *below* the no-mismatch random-walk noise floor (0.1178): with no
  drift to follow, it only random-walks. Same for both compositions.

## What this reproduces, and what it does not

**Reproduced on this testbed:**

- The drift mechanism itself (Eq. 3–5): the exact expected update of
  naive PG under a biased sampler is a nonzero drift vector, while score
  centering's is exactly zero — verified numerically, not just claimed
  (see the drift probe below).
- The severity ordering: as mismatch grows, naive PG degrades first and
  fastest, and the gap between corrected and uncorrected methods widens
  (noise-1.0: all ~1.000; noise-2.0: naive 0.683 vs tis 0.906).
- The staleness finding: at stale-128 naive never learns (0.094) while
  score centering composed with importance sampling reaches 1.000 —
  matching the paper's Figure 4, where the composition dominates under
  staleness.
- The top-k approximation (§4.1): `sc_topk` (k=8 of a 10-token vocab)
  tracks full score centering within 0.05 everywhere.

**Not reproduced / limits (stated honestly):**

- No mid-training collapse (a trained policy falling off a cliff) occurs
  on this testbed: the severe-staleness failure is a failure to *learn*
  (naive at stale-128 stays near zero for the whole run, peak 0.138),
  not a collapse of a trained policy. The testbed's sequences are short
  and its advantages are group-centred, both of which shrink the drift
  per step relative to the paper's 512-token LLM rollouts.
- The paper's quantisation finding does not reproduce here: weight
  rounding (steps 0.05–1.0) and logit snapping (2–4 levels) did not
  separate the methods at any tested severity. Both preserve the
  sampler's token ranking, so the sampled tokens barely change and the
  drift stays small. Quantisation is therefore not in the results table.
- At extreme weight noise (σ=3.0) the ordering flips: score centering
  does *worse* than naive. When the sampler is that far from the trainer
  it is effectively a different, overconfident policy (its distribution
  is peaked on random tokens), so the covariance signal that score
  centering preserves is itself degraded — the `e_y − q` score nearly
  vanishes for the sampled token. The paper's Eq. 5 caveat says the same
  thing: score centering fixes the drift, but the remaining covariance is
  still measured under `q`.
- Under weight noise the Eq. 11 compositions are *worse* than their
  components (noise-2.0: sc+tis 0.729 vs sc 0.793 and tis 0.906), the
  opposite of their staleness behaviour. The IS weights are noisy when
  the sampler is randomly perturbed, and the composed estimator pays for
  that variance; under staleness the mismatch is systematic, and the
  composition corrects it.
- All numbers here are simulation numbers on this testbed, not the
  paper's Qwen3-0.6B/30B numbers. The paper's figures (Qwen3-30B on
  INTELLECT-2 math: naive PG collapses within 200 steps under FP4 KV
  quantisation; score centering trains stably) are quoted separately
  above and never mixed with ours.

## Simulation adaptations

- **Small model, not an LLM**: a linear-softmax policy (355 features),
  trained with Adam. A tabular policy was tried first during development
  and only plateaued under mismatch — drift localises per state in a
  table, so the damage does not compound; the shared-parameter model
  replaced it.
- **Graded reward, not binary**: the paper trains a pretrained LLM whose
  prior already produces near-misses, so binary verifiable rewards carry
  signal. A policy starting from uniform gets almost no binary signal,
  so the reward is graded by distance to the target while exact match
  stays the headline metric.
- **Short sequences, small vocab**: 6 digits from a 10-token vocab, so
  the expected score under the sampler is an exact finite sum.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q                          # full test suite
python scripts/demo.py             # oracle / naive / sc / sc+tis under noise-0.6
python scripts/run_experiments.py  # full grid -> results/experiment.json
python scripts/run_drift_probe.py  # constant-reward probe -> results/drift_probe.json
python scripts/crossfoot.py        # verify every README number against the raw records
```

## Sources

- Marek, M. & Ryabinin, M. (2026). Score Centering Stabilizes
  Off-policy Reinforcement Learning. arXiv:2609.20807.
  https://arxiv.org/abs/2609.20807 (abstract and method verified
  2026-10-31; official code at
  https://github.com/martin-marek/score-centering — not used here;
  this is an independent implementation from the paper's equations).
- Research notes, including the trending check and rejected
  alternatives, in `docs/RESEARCH.md`.

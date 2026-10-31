# Research notes — score-centering-lab

## Trending check (2026-10-31, before any paper research)

GitHub Trending, daily and weekly, checked 2026-10-31 ~00:00 ET:

- **Daily**: thedotmack/claude-mem (agent memory), milind-soni/OpenMausBot,
  pbakaus/impeccable (design skill), CopilotKit/OpenBot (agent coworkers),
  expo/skills, macro-inc/macro, mattpocock/skills, HKUDS/DeepTutor
  (personalised tutoring platform), anthropics/skills, JetBrains/thinkrail,
  obra/superpowers, lightpanda-io/browser (headless browser for AI).
- **Weekly**: cloudflare/security-audit-skill (54,258 stars, +12,838/week),
  AgriciDaniel/claude-seo, pbakaus/impeccable, obra/superpowers,
  TauricResearch/TradingAgents (multi-agent LLM trading framework),
  affaan-m/ECC (agent harness optimiser), ruvnet/ruflo, anthropics/skills.

Same pattern as Days 5–6: the page is dominated by agent **skill packs,
prompt packs, and platforms** — markdown/config artefacts with no
benchmarkable code weakness to optimise. The few code repos are either
full platforms (DeepTutor, OpenBot, TradingAgents) with no single
measurable weakness a one-night CPU build could honestly improve, or
GPU-only inference engines (DeepInfer was already considered on Day 5).
**No trending repo was picked; falling back to the paper-based original**,
as the standing rule requires when no trending repo has a genuine,
measurable weakness.

## Paper chosen

**Marek, M. & Ryabinin, M. (Together AI), "Score Centering Stabilizes
Off-policy Reinforcement Learning", arXiv:2609.20807, submitted 17 Sep
2026.** Verified on the arXiv abstract page and the full HTML version on
2026-10-31. Also surfaced by AI Weekly's alert on 2026-10-31 and cited by
several October 2026 papers (TRACE, arXiv:2610.07767; "A Sign Law for
Entropy Drift", arXiv:2610.34044; KLPO), which is how it was found.

Why this paper:

- **Current and load-bearing**: training-inference mismatch (TIM) is the
  failure mode behind RL reward collapse in production LLM training; the
  paper gives a one-line additive fix with exact drift cancellation.
- **Genuinely new mechanism**: every existing correction in the paper's
  Table 1 is importance-sampling based (clipping/masking the ratio);
  score centering uses no ratio at all — it subtracts the sampler's
  expected score. That makes it implementable *exactly* in a small
  testbed, where the expectation over the next token is a finite sum.
- **Measurable claims**: (1) drift is the cause of instability and
  accumulates (Sec. 4–5.1); (2) under severe weight noise only score
  centering (alone or composed) trains stably (Fig. 3); (3) under
  quantisation SC alone is best, under staleness SC+IS is best (Fig. 4);
  (4) the top-k approximation (k=128, even k=32) matches full SC
  (Sec. 4.1); (5) group centering does *not* remove drift (Sec. 4).
  All five are testable in this lab.

## Method, as implemented (from the paper, not from its code)

- Policy gradient identity (Eq. 1): `∇E_p[R] = E_p[R · ∇log p(y)]`.
- Drift/signal split (Eq. 3): `E_q[R·s] = E_q[R]·s̄ + Cov_q(R, s)`,
  with `s̄ = E_q[s]` the drift. Drift is distillation toward the sampler.
- Score centering (Eq. 4–5): `s̃ = s − s̄`; then `E_q[R·s̃] = Cov_q(R, s)`.
  With a softmax policy, `s̄ = q − p` in logit space, so the centred
  score for sampled token `y` is `e_y − q` — exact, no approximation.
- Scalar-loss form (Eq. 8): `L = −R(log p_y − Σ_v sg[q_v − ρp_v] log p_v)`.
  Verified against the centred score by finite differences in
  `tests/test_corrections.py`.
- Top-k approximation (Sec. 4.1): sampler head exact on top-k, tail
  modelled with the trainer's shape rescaled to the sampler's tail mass
  (`ρ` = sampler tail mass / trainer tail mass). Here k=8 (vocab is 10).
- Shared objective (Sec. 5.1): REINFORCE with group-centred rewards;
  all methods share the sampler, trainer, optimiser, and advantages.
- Mismatch settings (Sec. 5.2–5.3): Gaussian weight noise on the sampler
  (σ scale), fixed-step weight quantisation (int8/fp4 analogue), and
  staleness (sampler refreshed every k steps).

## Development findings (pilots, not published results)

- **Tabular policy (tried first)**: only plateaued under mismatch —
  drift localises per state, so the damage does not compound. Replaced
  by the shared-parameter linear policy.
- **SGD too slow on the linear policy** (sparse one-hot features):
  replaced by Adam (lr 0.05), the standard optimiser for this kind of
  training. The drift probe still uses SGD, where the analysis is about
  gradients, not the optimiser.
- **Composition formula (caught before the final grid)**: the first
  implementation of SC+IS used `w_y * (s_y - E_q[s])` — the IS weight
  times the centred score. Reading Appendix A.2 (Eq. 11) showed the
  paper's composition is `w_y * s_y - E_q[w*s]`: subtract the expectation
  of the *weighted* score. The wrong version collapsed to 0.228 under
  stale-64; the correct version reaches 1.000 there and rescues the
  stale-128 collapse entirely. The TIS/MIS weights were also corrected
  to the paper's Table 1: TIS clips only at the top (`min(r, 2)`), MIS
  keeps the ratio itself inside [0.5, 5] and drops the token outside.
  All pilot TIS/MIS/composition numbers from before this fix are
  invalid and were discarded; only naive/SC/oracle pilot numbers
  survive (those methods were unchanged).
- **Quantisation does not separate on this testbed**: fixed-step weight
  rounding (steps 0.05–1.0) and logit snapping (2–4 levels) left every
  method at 1.000. Both preserve the sampler's token ranking, so the
  sampled tokens barely change. Weight noise and staleness reorder the
  distribution, which is what bites. Quantisation is therefore not in
  the final grid; the negative result is reported in the README.
- **Severity calibration** (2-seed pilots, Adam lr 0.05, 400 steps):
  noise σ=1.0 no separation; σ=1.5 SC/TIS ~0.96 vs naive 0.81;
  σ=2.0 IS methods ~0.86–0.87, SC 0.80, naive 0.72; σ=3.0 ordering
  flips (naive 0.54 > SC 0.37). Stale-64 mild (naive 0.97); stale-128
  naive collapses to 0.06 while SC holds 0.34 and every IS/composition
  method reaches 1.00.

## What was *not* copied

The paper's official code is at
https://github.com/martin-marek/score-centering (linked in the paper).
It was **not read or used**: every line here is an independent
implementation from the equations in the arXiv HTML version. The repo is
cited in the README only as the paper's official implementation.

## Alternatives considered and rejected

- **CMU unified training** (Song et al., arXiv:2610.???, 29 Oct 2026):
  one likelihood objective for pretrain/midtrain/SFT/RL. Very current and
  futuristic, but a faithful CPU reproduction needs a real small-model
  training run with a data-mix schedule — too large for one night to do
  honestly, and the paper's own results are mixed across budgets.
  Held as a candidate for a later night with a bigger scope.
- **Barron softmax fix** (arXiv:2610.31642, held from Day 6): needs a
  bf16 training harness to show anything; still held.
- **TradingAgents** (weekly trending): a trading framework — out of
  scope, and no single measurable weakness to optimise in a CPU sim.
- **DeepTutor / OpenBot / claude-mem**: full platforms, not
  benchmarkable in one night.

## Simulation adaptations (stated honestly in the README)

1. **Small model, not an LLM**: a linear-softmax policy over a
   355-dim one-hot state representation (remaining-sum × position pairs
   + target), trained with Adam. Parameters are shared across states, so
   drift propagates — the property that makes the paper's collapse
   possible. A tabular policy was tried first during development and only
   plateaued under mismatch (drift localises per state, so the damage
   does not compound); the shared-parameter model replaced it.
2. **Graded reward, not binary**: the paper trains a pretrained LLM
   whose prior already produces near-misses, so binary verifiable
   rewards carry signal. A policy starting from uniform gets almost no
   binary signal, so the reward is graded by distance to the target
   while exact match stays the headline metric.
3. **Short sequences, small vocab**: 6 digits from a 10-token vocab, so
   the expected score under the sampler is an exact finite sum and the
   top-k approximation can be checked against the exact version.
4. **All numbers in the README are simulation numbers** on this testbed,
   not the paper's Qwen3/Countdown numbers. The paper's figures are
   quoted separately, with their setting named, and never mixed with
   ours.

"""The constant-reward drift probe (paper Section 4, thought experiment).

With a constant reward ``R = +1`` for every rollout there is no learning
signal at all: on-policy, the expected gradient is exactly zero at every
prefix because ``E_p[s] = 0``. Under training-inference mismatch the naive
policy gradient has a nonzero expected gradient -- the drift
``E_q[R] * E_q[s]`` of Eq. 3 -- which pushes the trainer toward the sampler
at every step. Because the sampler is a biased copy of the trainer that is
re-derived from the trainer after each step, the error compounds in a
feedback loop instead of converging.

Two measurements:

1. **Exact expected update** (the headline): the expectation of each
   method's per-token gradient under the sampler, computed exactly by
   summing over the vocabulary -- no sampling noise. Score centering's is
   exactly zero (Eq. 5); naive's is the drift vector ``q - p`` (Eq. 3);
   TIS/MIS sit in between, because clipping/masking reintroduces drift
   (paper Section 4).

2. **Training dynamics**: constant-reward training tracks how the policy
   moves. Naive PG distills the trainer toward the sampler, so the KL
   between the policy and the *sampler* shrinks while the KL to the
   initialisation grows systematically; score centering only random-walks
   (its expected update is zero, so its movement matches the no-mismatch
   noise floor). The noise floor is reported from an exact-sampler control
   run, so the comparison is honest.
"""

from __future__ import annotations

import numpy as np

from .corrections import drift_norm, token_gradients
from .policy import LinearPolicy, SamplerView, features, softmax
from .task import MAX_PARTIAL, SEQ_LEN, TARGET_MIN, sample_prompts, state_ids_for_rollout


def probe_states(rng: np.random.Generator, n_prompts: int = 64) -> np.ndarray:
    """A fixed set of states spanning targets, positions, partial sums."""
    prompts = sample_prompts(n_prompts, rng)
    states = []
    for pos in range(SEQ_LEN):
        partial = rng.integers(0, 9 * pos + 1, size=n_prompts) if pos else np.zeros(n_prompts, dtype=int)
        target_idx = prompts - TARGET_MIN
        states.append((target_idx * SEQ_LEN + pos) * (MAX_PARTIAL + 1) + partial)
    return np.concatenate(states)


def expected_update(
    policy: LinearPolicy,
    sampler: SamplerView,
    method: str,
    states: np.ndarray,
    topk: int = 8,
) -> dict:
    """Exact expected per-token gradient under the sampler, averaged over
    the given states, with constant advantage +1.

    Returns the mean gradient vector (in logit space) and its L2 norm.
    Computed by summing over the vocabulary: E_q[g] = sum_v q_v * g(v).
    """
    p = policy.probs(states)
    q = sampler.probs(states)
    # E_q[g] = sum_v q_v * g(v): evaluate the gradient for every possible
    # sampled token v and weight by the sampler's probability of v.
    expected = np.zeros(p.shape[-1])
    for v in range(p.shape[-1]):
        grads = token_gradients(method, p, q, np.full(p.shape[0], v), np.ones(p.shape[0]), topk=topk)
        expected += (q[:, v][:, None] * grads).sum(axis=0)
    expected /= states.shape[0]
    return {"mean_gradient": expected, "norm": float(np.linalg.norm(expected))}


def kl_between(policy_a: LinearPolicy, policy_b_logits, states: np.ndarray) -> float:
    """Mean KL(policy_a || policy_b) over states; policy_b given by logits."""
    pa = policy_a.probs(states)
    pb = softmax(policy_b_logits(states))
    return float(
        np.sum(pa * (np.log(np.maximum(pa, 1e-12)) - np.log(np.maximum(pb, 1e-12))), axis=-1).mean()
    )


def run_drift_probe(
    method: str,
    sampler_factory,
    seed: int = 0,
    steps: int = 150,
    lr: float = 1.0,
    group: int = 8,
    batch_prompts: int = 48,
    topk: int = 8,
) -> dict:
    """Train with constant reward +1 and track policy movement."""
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(rng=rng)
    params0 = (policy.W.copy(), policy.b.copy())
    sampler = sampler_factory(policy)
    states_probe = probe_states(np.random.default_rng(999))

    initial_expected = expected_update(policy, sampler, method, states_probe, topk=topk)
    kl_to_init, kl_to_sampler, drift = [], [], []
    for step in range(steps):
        prompts = sample_prompts(batch_prompts, rng)
        # Rollouts from the sampler; rewards are constant, so the advantage
        # is +1 for every token (uncentred, as in the paper's Section 4).
        targets = np.repeat(prompts, group)
        digits = np.zeros((batch_prompts * group, SEQ_LEN), dtype=np.int64)
        partial = np.zeros(batch_prompts * group, dtype=np.int64)
        for pos in range(SEQ_LEN):
            target_idx = targets - TARGET_MIN
            states = (target_idx * SEQ_LEN + pos) * (MAX_PARTIAL + 1) + partial
            probs = sampler.probs(states)
            u = rng.random(probs.shape[0])
            token = (np.cumsum(probs, axis=-1) < u[:, None]).sum(axis=1)
            digits[:, pos] = token
            partial = partial + token
        states = state_ids_for_rollout(targets, digits)
        p = policy.probs(states)
        q = sampler.probs(states)
        advantages = np.ones(digits.shape)
        grads = token_gradients(method, p, q, digits, advantages, topk=topk)
        phi = features(states)
        n_rollouts = states.shape[0]
        dW = np.einsum("nld,nlk->dk", phi, grads) / n_rollouts
        db = grads.sum(axis=(0, 1)) / n_rollouts
        policy.W += lr * dW
        policy.b += lr * db
        sampler.sync(policy)
        if (step + 1) % 25 == 0 or step == 0:
            kl_to_init.append(
                kl_between(policy, lambda s: features(s) @ params0[0] + params0[1], states_probe)
            )
            kl_to_sampler.append(kl_between(policy, sampler.logits, states_probe))
            drift.append(drift_norm(p, q))
    final_expected = expected_update(policy, sampler, method, states_probe, topk=topk)
    return {
        "method": method,
        "seed": seed,
        "initial_expected_update_norm": initial_expected["norm"],
        "final_expected_update_norm": final_expected["norm"],
        "final_kl_to_init": kl_to_init[-1],
        "final_kl_to_sampler": kl_to_sampler[-1],
        "initial_kl_to_sampler": kl_to_sampler[0],
        "mean_drift_norm": float(np.mean(drift)),
        "kl_to_init": kl_to_init,
        "kl_to_sampler": kl_to_sampler,
        "drift_norm": drift,
    }

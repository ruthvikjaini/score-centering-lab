"""Training loop: rollouts from the sampler, gradients through the trainer.

One step: sample a batch of prompts, draw ``group`` rollouts per prompt
from the *sampler* (the biased inference engine), score them with the
verifiable reward, centre the rewards within each prompt group (the shared
objective of the paper's Section 5.1: REINFORCE with group-centred
rewards), and apply the correction method's per-token gradients to the
trainer's logits.
"""

from __future__ import annotations

import numpy as np

from .corrections import drift_norm, token_gradients
from .optim import Adam
from .policy import LinearPolicy, SamplerView, features
from .task import (
    SEQ_LEN,
    exact_match,
    graded_reward,
    sample_prompts,
    state_ids_for_rollout,
)


def rollout(
    sampler: SamplerView,
    prompts: np.ndarray,
    group: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``group`` rollouts per prompt from the sampler.

    Returns ``digits`` (B*G, SEQ_LEN), ``states`` (B*G, SEQ_LEN) and
    ``sums`` (B*G,).
    """
    batch = prompts.shape[0]
    targets = np.repeat(prompts, group)
    digits = np.zeros((batch * group, SEQ_LEN), dtype=np.int64)
    partial = np.zeros(batch * group, dtype=np.int64)
    for pos in range(SEQ_LEN):
        states = _states_for(targets, pos, partial)
        probs = sampler.probs(states)
        u = rng.random(probs.shape[0])
        token = (np.cumsum(probs, axis=-1) < u[:, None]).sum(axis=1)
        digits[:, pos] = token
        partial = partial + token
    states = state_ids_for_rollout(targets, digits)
    return digits, states, digits.sum(axis=1)


def _states_for(targets: np.ndarray, position: int, partial: np.ndarray) -> np.ndarray:
    from .task import MAX_PARTIAL, TARGET_MIN

    target_idx = targets - TARGET_MIN
    return (target_idx * SEQ_LEN + position) * (MAX_PARTIAL + 1) + partial


def group_advantages(rewards: np.ndarray, group: int) -> np.ndarray:
    """Centre rewards within each prompt's group (mean subtraction)."""
    grouped = rewards.reshape(-1, group)
    centred = grouped - grouped.mean(axis=1, keepdims=True)
    return centred.reshape(-1)


def train_step(
    policy: LinearPolicy,
    sampler: SamplerView,
    prompts: np.ndarray,
    method: str,
    lr: float,
    group: int,
    rng: np.random.Generator,
    topk: int = 8,
    optim: dict | None = None,
) -> dict:
    """One policy-gradient step. Returns step diagnostics.

    ``optim`` is ``{"W": Adam, "b": Adam}`` for Adam, or ``None`` for plain
    SGD with step size ``lr``. The gradient is the same either way; only
    the optimiser differs.
    """
    digits, states, sums = rollout(sampler, prompts, group, rng)
    targets = np.repeat(prompts, group)
    rewards = graded_reward(sums, targets)
    advantages = group_advantages(rewards, group)

    # Trainer and sampler distributions at every visited prefix.
    p = policy.probs(states)
    q = sampler.probs(states)
    grads = token_gradients(method, p, q, digits, advantages[:, None].repeat(SEQ_LEN, axis=1), topk=topk)
    # Per-token logit gradients -> shared-parameter gradients, averaged
    # over rollouts (mean loss, as in standard policy-gradient practice):
    # dW = sum_tokens phi(s) (x) grad_z, db = sum_tokens grad_z.
    phi = features(states)
    n_rollouts = states.shape[0]
    dW = np.einsum("nld,nlk->dk", phi, grads) / n_rollouts
    db = grads.sum(axis=(0, 1)) / n_rollouts
    if optim is None:
        policy.W += lr * dW
        policy.b += lr * db
    else:
        optim["W"].step(policy.W, dW)
        optim["b"].step(policy.b, db)
    sampler.sync(policy)

    return {
        "mean_reward": float(rewards.mean()),
        "exact_match": float(exact_match(sums, targets).mean()),
        "drift_norm": drift_norm(p, q),
    }


def evaluate(policy: LinearPolicy, prompts: np.ndarray) -> dict:
    """Greedy decoding on fresh prompts: exact-match accuracy + reward."""
    digits = np.zeros((prompts.shape[0], SEQ_LEN), dtype=np.int64)
    partial = np.zeros(prompts.shape[0], dtype=np.int64)
    for pos in range(SEQ_LEN):
        states = _states_for(prompts, pos, partial)
        token = policy.greedy(states)
        digits[:, pos] = token
        partial = partial + token
    sums = digits.sum(axis=1)
    return {
        "exact_match": float(exact_match(sums, prompts).mean()),
        "mean_reward": float(graded_reward(sums, prompts).mean()),
    }


def run_training(
    method: str,
    sampler_factory,
    seed: int = 0,
    steps: int = 300,
    lr: float = 0.05,
    group: int = 8,
    batch_prompts: int = 24,
    eval_every: int = 25,
    topk: int = 8,
    optimizer: str = "adam",
) -> dict:
    """Train one (method, mismatch) configuration. Returns the run record.

    ``sampler_factory`` takes the fresh policy and returns a sampler view.
    The oracle method always uses an exact sampler regardless of factory.
    ``optimizer`` is "adam" (default, lr is the Adam step size) or "sgd".
    """
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(rng=rng)
    if method == "oracle":
        from .policy import ExactSampler

        sampler = ExactSampler(policy)
    else:
        sampler = sampler_factory(policy)
    optim = None
    if optimizer == "adam":
        optim = {"W": Adam(policy.W.shape, lr=lr), "b": Adam(policy.b.shape, lr=lr)}
    elif optimizer != "sgd":
        raise ValueError(f"unknown optimizer {optimizer!r}")

    eval_prompts = sample_prompts(200, np.random.default_rng(10_000 + seed))
    history = {"eval_exact_match": [], "eval_mean_reward": [], "drift_norm": []}
    eval_steps = []
    for step in range(steps):
        prompts = sample_prompts(batch_prompts, rng)
        diag = train_step(policy, sampler, prompts, method, lr, group, rng, topk=topk, optim=optim)
        if (step + 1) % eval_every == 0 or step == 0:
            ev = evaluate(policy, eval_prompts)
            history["eval_exact_match"].append(ev["exact_match"])
            history["eval_mean_reward"].append(ev["mean_reward"])
            history["drift_norm"].append(diag["drift_norm"])
            eval_steps.append(step + 1)
    final = history["eval_exact_match"][-1]
    peak = max(history["eval_exact_match"])
    steps_to_080 = next(
        (s for s, a in zip(eval_steps, history["eval_exact_match"]) if a >= 0.80),
        None,
    )
    return {
        "method": method,
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "optimizer": optimizer,
        "final_exact_match": final,
        "peak_exact_match": peak,
        "final_mean_reward": history["eval_mean_reward"][-1],
        "mean_drift_norm": float(np.mean(history["drift_norm"])),
        "collapsed": bool(final < 0.40 and peak >= 0.60),
        "steps_to_080": steps_to_080,
        "eval_steps": eval_steps,
        "history": history,
    }

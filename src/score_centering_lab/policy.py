"""Linear-softmax policy and sampler views.

The policy is a *shared-parameter* model, as in a real network: logits are
a linear function of state features,

    z(s, a) = w_a . phi(s) + b_a,

with ``phi(s)`` a one-hot feature vector over (remaining sum, position,
target). Parameters are shared across states, so a biased update in one
state moves the weights for all states -- the channel through which
training-inference mismatch compounds in a real model. A purely tabular
policy localises the damage per state and, in our pilots, only plateaued
under mismatch instead of collapsing; the shared-parameter model is the
faithful analogue of the paper's weight-space mismatch.

The trainer owns ``(W, b)``. A *sampler view* returns the (weight, bias)
the rollout engine actually uses, which may differ from the trainer's:
that difference is the training-inference mismatch (TIM). Weight noise,
weight quantisation, and staleness are all applied in *weight* space,
exactly as in Marek & Ryabinin Sections 2.1 and 5.
"""

from __future__ import annotations

import numpy as np

from .task import (
    MAX_PARTIAL,
    N_ACTIONS,
    N_STATES,
    SEQ_LEN,
    TARGET_MAX,
    TARGET_MIN,
)

# Feature layout: one-hot(remaining sum, position) pairs ++ one-hot(target).
# The (remaining, position) pair is the quantity the optimal policy needs --
# which digit to play depends on how much is left *and* how many positions
# remain -- while the weights are still shared across targets and states,
# so a biased update in one state moves the policy everywhere.
N_REMAINING = MAX_PARTIAL + 1  # 55
N_PAIRS = N_REMAINING * SEQ_LEN  # 330
N_FEATURES = N_PAIRS + (TARGET_MAX - TARGET_MIN + 1)  # 330+25 = 355


def decode_state(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """State ids -> (target, position, partial_sum)."""
    ids = np.asarray(ids)
    partial = ids % (MAX_PARTIAL + 1)
    rest = ids // (MAX_PARTIAL + 1)
    position = rest % SEQ_LEN
    target_idx = rest // SEQ_LEN
    return target_idx + TARGET_MIN, position, partial


_FEATURE_MATRIX: np.ndarray | None = None


def feature_matrix() -> np.ndarray:
    """One-hot features for every state, built once and cached.

    Shape (N_STATES, N_FEATURES). States are dense (8,250 of them), so
    the full matrix is small (~23 MB) and indexing it beats rebuilding
    one-hots on every call.
    """
    global _FEATURE_MATRIX
    if _FEATURE_MATRIX is None:
        ids = np.arange(N_STATES)
        target, position, partial = decode_state(ids)
        remaining = np.clip(target - partial, 0, MAX_PARTIAL)
        mat = np.zeros((N_STATES, N_FEATURES), dtype=np.float64)
        idx = np.arange(N_STATES)
        pair = remaining * SEQ_LEN + position
        mat[idx, pair] = 1.0
        mat[idx, N_PAIRS + (target - TARGET_MIN)] = 1.0
        _FEATURE_MATRIX = mat
    return _FEATURE_MATRIX


def features(ids: np.ndarray) -> np.ndarray:
    """One-hot feature matrix for state ids. Shape (..., N_FEATURES)."""
    return feature_matrix()[np.asarray(ids)]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax, numerically stable."""
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class LinearPolicy:
    """Shared-parameter softmax policy: logits = phi(s) W + b."""

    def __init__(
        self,
        W: np.ndarray | None = None,
        b: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng(0)
        if W is None:
            # Small init keeps the initial policy close to uniform.
            W = 0.05 * rng.standard_normal((N_FEATURES, N_ACTIONS))
        if b is None:
            b = np.zeros(N_ACTIONS)
        self.W = np.array(W, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)

    def copy(self) -> "LinearPolicy":
        return LinearPolicy(W=self.W.copy(), b=self.b.copy())

    def params_vector(self) -> np.ndarray:
        return np.concatenate([self.W.reshape(-1), self.b])

    def logits(self, states: np.ndarray) -> np.ndarray:
        return features(states) @ self.W + self.b

    def probs(self, states: np.ndarray) -> np.ndarray:
        return softmax(self.logits(states))

    def log_probs(self, states: np.ndarray) -> np.ndarray:
        return log_softmax(self.logits(states))

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self.probs(states)
        u = rng.random(p.shape[:-1] + (1,))
        return (np.cumsum(p, axis=-1) < u).sum(axis=-1)

    def greedy(self, states: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(states), axis=-1)


class SamplerView:
    """The (W, b) the rollout engine samples from.

    ``logits(states)`` returns sampler logits; ``sync(policy)`` is called
    after each trainer step so views that track the trainer (quantisation,
    staleness) can refresh.
    """

    def logits(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def probs(self, states: np.ndarray) -> np.ndarray:
        return softmax(self.logits(states))

    def sync(self, policy: LinearPolicy) -> None:
        pass


class ExactSampler(SamplerView):
    """No mismatch: sampler and trainer are the same model (oracle)."""

    def __init__(self, policy: LinearPolicy):
        self.policy = policy

    def logits(self, states: np.ndarray) -> np.ndarray:
        return self.policy.logits(states)


class WeightNoiseSampler(SamplerView):
    """Sampler weights = trainer weights + fixed Gaussian offset.

    Paper Section 5.2: ``theta_sampler = theta_trainer + Delta theta``,
    with ``Delta theta`` sampled once at the start of training and held
    fixed. The offset is the least realistic mismatch source but gives a
    continuous scale of TIM and the fastest collapse, so it is the paper's
    tool for initial experiments.
    """

    def __init__(self, policy: LinearPolicy, sigma: float, rng: np.random.Generator):
        self.policy = policy
        self.sigma = float(sigma)
        self.delta_W = sigma * rng.standard_normal(policy.W.shape)
        self.delta_b = sigma * rng.standard_normal(policy.b.shape)

    def logits(self, states: np.ndarray) -> np.ndarray:
        return features(states) @ (self.policy.W + self.delta_W) + (self.policy.b + self.delta_b)


class QuantizedSampler(SamplerView):
    """Sampler weights quantised onto a fixed low-precision grid.

    Paper Section 5.3: the trainer stays in full precision while the
    sampler (weights, activations, KV cache) is quantised. The grid step is
    fixed, mimicking a fixed-point inference engine whose step is set by
    the weight scale rather than by each row's range; the grid is
    recomputed from the current trainer weights on every call, so the
    sampler tracks the trainer up to rounding.
    """

    def __init__(self, policy: LinearPolicy, step: float = 0.1):
        self.policy = policy
        self.step = float(step)

    def logits(self, states: np.ndarray) -> np.ndarray:
        Wq = np.round(self.policy.W / self.step) * self.step
        bq = np.round(self.policy.b / self.step) * self.step
        return features(states) @ Wq + bq


class StaleSampler(SamplerView):
    """Sampler is a stale snapshot of the trainer (paper Section 5.3).

    The inference engine is refreshed only every ``sync_every`` trainer
    steps, so rollouts between syncs come from an older checkpoint. This is
    the mismatch source of disaggregated, asynchronous RL.
    """

    def __init__(self, policy: LinearPolicy, sync_every: int = 32):
        self.policy = policy
        self.sync_every = int(sync_every)
        self.W_snap = policy.W.copy()
        self.b_snap = policy.b.copy()
        self.steps_since_sync = 0

    def logits(self, states: np.ndarray) -> np.ndarray:
        return features(states) @ self.W_snap + self.b_snap

    def sync(self, policy: LinearPolicy) -> None:
        self.steps_since_sync += 1
        if self.steps_since_sync >= self.sync_every:
            self.W_snap = policy.W.copy()
            self.b_snap = policy.b.copy()
            self.steps_since_sync = 0


class LogitLevelsSampler(SamplerView):
    """Sampler logits snapped onto a few discrete levels (severe quantisation).

    The trainer's logits are rounded onto ``levels`` evenly spaced values
    spanning the per-state logit range, mimicking a very low-precision
    inference engine (2-bit style). Unlike rounding the *weights* onto a
    fixed grid -- which becomes negligible once the weights grow large --
    snapping the logits directly changes the sampler's distribution at
    every scale: with 2 levels the sampler is uniform over the trainer's
    argmax tokens and near-zero elsewhere.
    """

    def __init__(self, policy: LinearPolicy, levels: int = 4):
        self.policy = policy
        self.levels = int(levels)

    def logits(self, states: np.ndarray) -> np.ndarray:
        z = self.policy.logits(states)
        lo = z.min(axis=-1, keepdims=True)
        hi = z.max(axis=-1, keepdims=True)
        span = np.maximum(hi - lo, 1e-12)
        step = span / (self.levels - 1)
        return lo + np.round((z - lo) / step) * step

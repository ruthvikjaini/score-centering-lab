"""The target-sum task: a small verifiable-reward sequence task.

Each prompt is a target integer. The policy generates a sequence of
``SEQ_LEN`` digits (0-9); the reward is graded by how close the digit sum
is to the target, and the headline metric is exact-match accuracy (sum ==
target). The state the policy sees is ``(target, position, partial_sum)``,
so a tabular policy can in principle learn to hit the target exactly.

Why a graded reward instead of the paper's binary one: the paper trains a
pretrained LLM whose prior already produces near-miss answers, so a binary
verifiable reward carries signal. A tabular policy starts uniform, where a
binary reward is almost always 0 and group-centred advantages vanish.
The graded reward keeps the same verifiable structure (exact match is the
success criterion) while giving a dense learning signal. The drift
mechanism under study -- a biased sampler pulling the trainer -- does not
depend on the reward shape; the constant-reward drift probe in
``drift.py`` isolates it with no reward signal at all.
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 6
N_ACTIONS = 10  # digits 0..9
TARGET_MIN = 12
TARGET_MAX = 36
N_TARGETS = TARGET_MAX - TARGET_MIN + 1  # 25
MAX_PARTIAL = 9 * SEQ_LEN  # 54
# State id = ((target_idx * SEQ_LEN) + position) * (MAX_PARTIAL + 1) + partial
N_STATES = N_TARGETS * SEQ_LEN * (MAX_PARTIAL + 1)

# Graded-reward scale: a miss of REWARD_SCALE or more scores 0.
REWARD_SCALE = 18.0


def state_id(target: int, position: int, partial_sum: int) -> int:
    target_idx = target - TARGET_MIN
    return (target_idx * SEQ_LEN + position) * (MAX_PARTIAL + 1) + partial_sum


def state_ids_for_rollout(targets: np.ndarray, digits: np.ndarray) -> np.ndarray:
    """State id at each position of each rollout.

    ``targets`` shape (B,), ``digits`` shape (B, SEQ_LEN); returns (B, SEQ_LEN).
    """
    batch = targets.shape[0]
    positions = np.arange(SEQ_LEN)[None, :].repeat(batch, axis=0)
    partial = np.zeros((batch, SEQ_LEN), dtype=np.int64)
    partial[:, 1:] = np.cumsum(digits[:, :-1], axis=1)
    target_idx = (targets - TARGET_MIN)[:, None]
    return (target_idx * SEQ_LEN + positions) * (MAX_PARTIAL + 1) + partial


def graded_reward(sums: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Dense verifiable reward in [0, 1]: 1 iff the sum hits the target."""
    return np.maximum(0.0, 1.0 - np.abs(sums - targets) / REWARD_SCALE)


def exact_match(sums: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (sums == targets).astype(np.float64)


def sample_prompts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n target prompts uniformly."""
    return rng.integers(TARGET_MIN, TARGET_MAX + 1, size=n)

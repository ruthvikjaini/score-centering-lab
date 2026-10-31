"""Tests for the policy: features, softmax, sampler views."""

import numpy as np
import pytest

from score_centering_lab.policy import (
    LinearPolicy,
    QuantizedSampler,
    StaleSampler,
    WeightNoiseSampler,
    decode_state,
    features,
    softmax,
)
from score_centering_lab.task import N_STATES, SEQ_LEN, state_id


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 10))
    np.testing.assert_allclose(softmax(z).sum(axis=-1), 1.0, atol=1e-12)


def test_features_are_one_hot_pairs():
    ids = np.array([state_id(20, 0, 0), state_id(20, 3, 17), state_id(36, 5, 54)])
    phi = features(ids)
    # Each row has exactly two ones: (remaining, position) pair + target.
    assert (phi.sum(axis=-1) == 2.0).all()
    assert set(np.unique(phi)) <= {0.0, 1.0}


def test_decode_state_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        target = int(rng.integers(12, 37))
        pos = int(rng.integers(0, SEQ_LEN))
        partial = int(rng.integers(0, 55))
        t, p, s = decode_state(np.array([state_id(target, pos, partial)]))
        assert (t[0], p[0], s[0]) == (target, pos, partial)


def test_policy_probs_are_distributions():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    ids = np.arange(0, N_STATES, 997)
    np.testing.assert_allclose(policy.probs(ids).sum(axis=-1), 1.0, atol=1e-12)


def test_policy_greedy_matches_argmax_logits():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    ids = np.arange(0, 500)
    np.testing.assert_array_equal(policy.greedy(ids), np.argmax(policy.logits(ids), axis=-1))


def test_exact_sampler_matches_policy():
    from score_centering_lab.policy import ExactSampler

    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = ExactSampler(policy)
    ids = np.arange(0, 500)
    np.testing.assert_allclose(sampler.logits(ids), policy.logits(ids), atol=1e-12)


def test_weight_noise_sampler_is_biased_copy():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    rng = np.random.default_rng(5)
    sampler = WeightNoiseSampler(policy, sigma=0.3, rng=rng)
    ids = np.arange(0, 200)
    # Sampler logits differ from trainer logits (sigma > 0).
    assert np.abs(sampler.logits(ids) - policy.logits(ids)).max() > 0.1
    # The offset is fixed: two calls give identical logits.
    first = sampler.logits(ids)
    np.testing.assert_allclose(sampler.logits(ids), first, atol=0)
    # And it tracks the trainer: shifting every weight by 1.0 shifts every
    # logit by 2.0 (each state has exactly two active features).
    policy.W += 1.0
    np.testing.assert_allclose(sampler.logits(ids), first + 2.0, atol=1e-12)


def test_weight_noise_zero_sigma_is_exact():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = WeightNoiseSampler(policy, sigma=0.0, rng=np.random.default_rng(0))
    ids = np.arange(0, 200)
    np.testing.assert_allclose(sampler.logits(ids), policy.logits(ids), atol=1e-12)


def test_quantized_sampler_rounds_weights():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = QuantizedSampler(policy, step=0.5)
    ids = np.arange(0, 200)
    logits = sampler.logits(ids)
    # Logits are phi @ Wq + bq with Wq, bq on a 0.5 grid: differences
    # between two rows sharing the same features are multiples of 0.5.
    # Direct check: recompute from rounded weights.
    Wq = np.round(policy.W / 0.5) * 0.5
    bq = np.round(policy.b / 0.5) * 0.5
    np.testing.assert_allclose(logits, features(ids) @ Wq + bq, atol=1e-12)


def test_stale_sampler_lags_then_syncs():
    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = StaleSampler(policy, sync_every=3)
    ids = np.arange(0, 100)
    before = sampler.logits(ids)
    policy.W += 0.7
    # Not yet synced: sampler still serves the old snapshot.
    sampler.sync(policy)
    np.testing.assert_allclose(sampler.logits(ids), before, atol=1e-12)
    sampler.sync(policy)
    np.testing.assert_allclose(sampler.logits(ids), before, atol=1e-12)
    # Third sync refreshes the snapshot.
    sampler.sync(policy)
    np.testing.assert_allclose(sampler.logits(ids), policy.logits(ids), atol=1e-12)


def test_logit_levels_sampler_snaps_to_levels():
    from score_centering_lab.policy import LogitLevelsSampler

    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = LogitLevelsSampler(policy, levels=4)
    ids = np.arange(0, 200)
    z = policy.logits(ids)
    snapped = sampler.logits(ids)
    lo = z.min(axis=-1, keepdims=True)
    hi = z.max(axis=-1, keepdims=True)
    step = (hi - lo) / 3
    np.testing.assert_allclose(snapped, lo + np.round((z - lo) / step) * step, atol=1e-12)
    # With 2 levels the sampler only ever emits the per-state min or max.
    sampler2 = LogitLevelsSampler(policy, levels=2)
    snapped2 = sampler2.logits(ids)
    # Direct check: every snapped value equals lo or hi for its row.
    is_lo = np.isclose(snapped2, lo, atol=1e-9)
    is_hi = np.isclose(snapped2, hi, atol=1e-9)
    assert (is_lo | is_hi).all()

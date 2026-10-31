"""Tests for the training loop and the drift probe."""

import numpy as np

from score_centering_lab.drift import run_drift_probe
from score_centering_lab.policy import ExactSampler, WeightNoiseSampler
from score_centering_lab.train import evaluate, group_advantages, rollout, run_training


def test_group_advantages_sum_to_zero_per_prompt():
    rng = np.random.default_rng(0)
    rewards = rng.random(40)
    adv = group_advantages(rewards, group=8)
    assert adv.shape == rewards.shape
    np.testing.assert_allclose(adv.reshape(-1, 8).sum(axis=1), 0.0, atol=1e-12)


def test_rollout_shapes_and_sums():
    from score_centering_lab.policy import LinearPolicy

    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = ExactSampler(policy)
    prompts = np.array([15, 20, 25])
    digits, states, sums = rollout(sampler, prompts, group=4, rng=np.random.default_rng(0))
    assert digits.shape == (12, 6)
    assert states.shape == (12, 6)
    np.testing.assert_array_equal(sums, digits.sum(axis=1))
    assert ((digits >= 0) & (digits <= 9)).all()


def test_oracle_learns_the_task():
    """Sanity: with no mismatch the policy must learn to hit targets."""
    rec = run_training("oracle", lambda p: ExactSampler(p), seed=0, steps=300, lr=0.05)
    assert rec["final_exact_match"] >= 0.6, rec


def test_evaluate_greedy_is_deterministic():
    from score_centering_lab.policy import LinearPolicy

    policy = LinearPolicy(rng=np.random.default_rng(0))
    prompts = np.array([15, 20, 25, 30])
    first = evaluate(policy, prompts)
    second = evaluate(policy, prompts)
    assert first == second


def test_drift_probe_expected_update():
    """Paper Eq. 3 vs Eq. 5, computed exactly: under a biased sampler the
    naive expected update is the nonzero drift vector, while score
    centering's expected update is exactly zero."""
    from score_centering_lab.drift import expected_update, probe_states
    from score_centering_lab.policy import LinearPolicy, WeightNoiseSampler

    policy = LinearPolicy(rng=np.random.default_rng(0))
    sampler = WeightNoiseSampler(policy, sigma=0.6, rng=np.random.default_rng(7))
    states = probe_states(np.random.default_rng(999))
    naive = expected_update(policy, sampler, "naive", states)
    sc = expected_update(policy, sampler, "sc", states)
    assert naive["norm"] > 0.05, naive
    assert sc["norm"] < 1e-12, sc
    # TIS/MIS shrink the drift but do not remove it (clipping/masking
    # reintroduces drift, paper Section 4).
    tis = expected_update(policy, sampler, "tis", states)
    assert tis["norm"] > 0.0, tis


def test_drift_probe_sc_stays_near_noise_floor():
    """Under constant reward, naive PG is dragged systematically toward the
    sampler (KL to the sampler shrinks, KL to init grows), while score
    centering only random-walks: its KL to init matches the no-mismatch
    noise floor."""
    factory = lambda p: WeightNoiseSampler(p, sigma=0.6, rng=np.random.default_rng(7))
    naive = run_drift_probe("naive", factory, seed=0, steps=150, lr=1.0)
    sc = run_drift_probe("sc", factory, seed=0, steps=150, lr=1.0)
    floor = run_drift_probe("naive", lambda p: ExactSampler(p), seed=0, steps=150, lr=1.0)
    # Naive distills toward the sampler: KL(policy || sampler) falls.
    assert naive["final_kl_to_sampler"] < naive["initial_kl_to_sampler"], naive
    # Score centering does not: its KL to the sampler stays put, and its
    # movement from init is within 2x of the no-mismatch random-walk floor.
    assert sc["final_kl_to_sampler"] >= 0.8 * sc["initial_kl_to_sampler"], sc
    assert sc["final_kl_to_init"] < 2.0 * floor["final_kl_to_init"] + 0.05, (sc, floor)
    assert naive["final_kl_to_init"] > 3.0 * floor["final_kl_to_init"], (naive, floor)


def test_run_training_record_shape():
    rec = run_training("naive", lambda p: ExactSampler(p), seed=1, steps=50, lr=0.05)
    assert set(rec) >= {
        "method", "seed", "final_exact_match", "peak_exact_match",
        "final_mean_reward", "mean_drift_norm", "collapsed", "steps_to_080",
    }
    assert 0.0 <= rec["final_exact_match"] <= 1.0
    assert len(rec["history"]["eval_exact_match"]) == len(rec["eval_steps"])

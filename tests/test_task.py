"""Tests for the target-sum task."""

import numpy as np

from score_centering_lab.task import (
    SEQ_LEN,
    exact_match,
    graded_reward,
    sample_prompts,
    state_id,
    state_ids_for_rollout,
)


def test_state_ids_encode_rollout():
    targets = np.array([20, 30])
    digits = np.array([[1, 2, 3, 4, 5, 5], [9, 9, 9, 9, 9, 9]])
    states = state_ids_for_rollout(targets, digits)
    assert states.shape == (2, SEQ_LEN)
    # Position 0 partial sum is 0; position 1 partial is the first digit.
    assert states[0, 0] == state_id(20, 0, 0)
    assert states[0, 1] == state_id(20, 1, 1)
    assert states[0, 2] == state_id(20, 2, 3)
    assert states[1, 5] == state_id(30, 5, 45)


def test_graded_reward_is_one_on_exact_match():
    targets = np.array([20, 25, 36])
    sums = np.array([20, 25, 36])
    np.testing.assert_allclose(graded_reward(sums, targets), 1.0)


def test_graded_reward_decays_with_miss():
    targets = np.array([20])
    assert graded_reward(np.array([20]), targets)[0] == 1.0
    assert graded_reward(np.array([29]), targets)[0] == 0.5  # miss 9 of 18
    assert graded_reward(np.array([38]), targets)[0] == 0.0  # miss >= 18
    assert graded_reward(np.array([2]), targets)[0] == 0.0


def test_exact_match_binary():
    targets = np.array([20, 20])
    sums = np.array([20, 21])
    np.testing.assert_array_equal(exact_match(sums, targets), [1.0, 0.0])


def test_sample_prompts_in_range():
    prompts = sample_prompts(1000, np.random.default_rng(0))
    assert prompts.min() >= 12 and prompts.max() <= 36
    assert len(np.unique(prompts)) > 20

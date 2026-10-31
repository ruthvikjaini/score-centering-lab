"""Tests for the correction methods: exact identities from the paper."""

import numpy as np
import pytest

from score_centering_lab.corrections import (
    METHODS,
    drift_norm,
    expected_score,
    token_gradients,
    topk_sampler_approx,
)


@pytest.fixture
def dists():
    rng = np.random.default_rng(0)
    p = rng.random((5, 10)) + 0.1
    p /= p.sum(axis=-1, keepdims=True)
    q = rng.random((5, 10)) + 0.1
    q /= q.sum(axis=-1, keepdims=True)
    y = rng.integers(0, 10, size=5)
    adv = rng.standard_normal(5)
    return p, q, y, adv


def test_expected_score_is_q_minus_p(dists):
    p, q, _, _ = dists
    np.testing.assert_allclose(expected_score(p, q), q - p, atol=1e-12)


def test_expected_score_zero_on_policy(dists):
    p, _, _, _ = dists
    np.testing.assert_allclose(expected_score(p, p), 0.0, atol=1e-12)


def test_score_centering_has_zero_expectation_under_sampler(dists):
    """Eq. 5: E_q[centred score] = 0 exactly, for any q != p.

    Checked by evaluating the centred score for every possible sampled
    token v and weighting by q_v: sum_v q_v (e_v - q) = q - q = 0.
    """
    p, q, _, _ = dists
    expected = np.zeros(p.shape[-1])
    for v in range(p.shape[-1]):
        grads = token_gradients("sc", p, q, np.full(p.shape[0], v), np.ones(p.shape[0]))
        expected += (q[:, v][:, None] * grads).sum(axis=0)
    np.testing.assert_allclose(expected, np.zeros(p.shape[-1]), atol=1e-12)


def test_naive_gradient_is_advantage_times_score(dists):
    p, q, y, adv = dists
    grads = token_gradients("naive", p, q, y, adv)
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[:, None], 1.0, axis=-1)
    np.testing.assert_allclose(grads, adv[:, None] * (one_hot - p), atol=1e-12)


def test_sc_gradient_is_advantage_times_centered_score(dists):
    p, q, y, adv = dists
    grads = token_gradients("sc", p, q, y, adv)
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[:, None], 1.0, axis=-1)
    np.testing.assert_allclose(grads, adv[:, None] * (one_hot - q), atol=1e-12)


def test_sc_is_noop_on_policy(dists):
    """Paper Section 4: with no mismatch, score centering is a no-op."""
    p, _, y, adv = dists
    np.testing.assert_allclose(
        token_gradients("sc", p, p, y, adv),
        token_gradients("naive", p, p, y, adv),
        atol=1e-12,
    )


def test_sc_matches_eq8_loss_gradient():
    """The Eq. 8 scalar loss has the centred score as its gradient.

    L = -R (log p_y - sum_v sg[q_v - rho p_v] log p_v), rho = 1 with the
    full vocabulary (no tail). dL/dz = -R ((e_y - p) - (q - p)) = -R(e_y - q).
    Checked by finite differences on a small logits vector.
    """
    rng = np.random.default_rng(3)
    z = rng.standard_normal(6)
    logits = z.copy()
    p = np.exp(z - z.max())
    p /= p.sum()
    q = rng.random(6) + 0.1
    q /= q.sum()
    y, R = 2, 0.7
    # The coefficient (q - rho*p) carries a stop-gradient in Eq. 8, so it
    # is frozen at the base point while differentiating.
    coef = q - p

    def loss(zz):
        pp = np.exp(zz - zz.max())
        pp /= pp.sum()
        logp = np.log(pp)
        correction = np.sum(coef * logp)
        return -R * (logp[y] - correction)

    eps = 1e-6
    numeric = np.zeros(6)
    for i in range(6):
        zp = logits.copy()
        zp[i] += eps
        zm = logits.copy()
        zm[i] -= eps
        numeric[i] = (loss(zp) - loss(zm)) / (2 * eps)
    one_hot = np.zeros(6)
    one_hot[y] = 1.0
    analytic = -R * (one_hot - q)  # gradient of L; ascent grad is -analytic
    np.testing.assert_allclose(numeric, analytic, atol=1e-6)


def test_tis_clips_ratio_at_top_only(dists):
    """Paper Table 1 (CISPO): TIS clips the ratio at 2, with no lower clip."""
    p, q, y, adv = dists
    grads = token_gradients("tis", p, q, y, adv)
    p_y = np.take_along_axis(p, y[:, None], -1)[..., 0]
    q_y = np.take_along_axis(q, y[:, None], -1)[..., 0]
    ratio = np.minimum(p_y / q_y, 2.0)
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[:, None], 1.0, axis=-1)
    np.testing.assert_allclose(grads, (adv * ratio)[:, None] * (one_hot - p), atol=1e-12)


def test_mis_keeps_ratio_in_band_drops_outside(dists):
    """Paper Table 1 (IcePop) + App. A.3 code: weight is the ratio itself
    inside [0.5, 5], and 0 outside (the token's gradient is dropped)."""
    p, q, y, adv = dists
    grads = token_gradients("mis", p, q, y, adv)
    p_y = np.take_along_axis(p, y[:, None], -1)[..., 0]
    q_y = np.take_along_axis(q, y[:, None], -1)[..., 0]
    ratio = p_y / q_y
    weight = np.where((ratio >= 0.5) & (ratio <= 5.0), ratio, 0.0)
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[:, None], 1.0, axis=-1)
    np.testing.assert_allclose(grads, (adv * weight)[:, None] * (one_hot - p), atol=1e-12)


def test_composition_expected_update_is_zero(dists):
    """Paper App. A.2, Eq. 11: the composed update has exactly zero
    expectation under the sampler for any weighting function, because it
    subtracts E_q[w*s], not w_y times the drift."""
    p, q, _, _ = dists
    for method in ("sc+tis", "sc+mis"):
        expected = np.zeros(p.shape[-1])
        for v in range(p.shape[-1]):
            grads = token_gradients(method, p, q, np.full(p.shape[0], v), np.ones(p.shape[0]))
            expected += (q[:, v][:, None] * grads).sum(axis=0)
        np.testing.assert_allclose(expected, np.zeros(p.shape[-1]), atol=1e-12)


def test_composition_matches_eq11_formula(dists):
    """sc+tis gradient = w_y * s_y - E_q[w*s], computed directly."""
    p, q, y, adv = dists
    grads = token_gradients("sc+tis", p, q, y, adv)
    ratio = p / np.maximum(q, 1e-12)
    w = np.minimum(ratio, 2.0)
    w_y = np.take_along_axis(w, y[:, None], -1)[..., 0]
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[:, None], 1.0, axis=-1)
    c = q * w
    expected_weighted = c - c.sum(axis=-1, keepdims=True) * p
    want = (adv * w_y)[:, None] * (one_hot - p) - adv[:, None] * expected_weighted
    np.testing.assert_allclose(grads, want, atol=1e-12)


def test_composition_is_noop_on_policy(dists):
    """With no mismatch, w = 1 and E_q[w*s] = 0, so the composition is
    plain policy gradient."""
    p, _, y, adv = dists
    for method in ("sc+tis", "sc+mis"):
        np.testing.assert_allclose(
            token_gradients(method, p, p, y, adv),
            token_gradients("naive", p, p, y, adv),
            atol=1e-12,
        )


def test_exact_is_has_zero_expected_weighted_score(dists):
    """Paper A.2 sanity check: with w = r exactly, E_q[w*s] = 0, so there
    is nothing left to centre -- the composition reduces to exact IS."""
    p, q, _, _ = dists
    ratio = p / np.maximum(q, 1e-12)
    c = q * ratio  # = p
    expected_weighted = c - c.sum(axis=-1, keepdims=True) * p
    np.testing.assert_allclose(expected_weighted, 0.0, atol=1e-12)


def test_topk_approx_is_valid_distribution(dists):
    p, q, _, _ = dists
    for k in (1, 4, 8, 10):
        qt = topk_sampler_approx(p, q, k)
        np.testing.assert_allclose(qt.sum(axis=-1), 1.0, atol=1e-12)
        assert (qt >= 0).all()


def test_topk_approx_matches_full_when_k_is_vocab(dists):
    p, q, _, _ = dists
    np.testing.assert_allclose(topk_sampler_approx(p, q, 10), q, atol=1e-12)


def test_topk_approx_keeps_head_exact(dists):
    p, q, _, _ = dists
    qt = topk_sampler_approx(p, q, 4)
    top_idx = np.argsort(-q, axis=-1)[:, :4]
    for row in range(5):
        np.testing.assert_allclose(qt[row, top_idx[row]], q[row, top_idx[row]], atol=1e-12)


def test_sc_topk_close_to_sc_for_peaked_sampler(dists):
    p, _, y, adv = dists
    # Peaked sampler: top-4 carries ~all the mass, so the approx is tight.
    q = np.full((5, 10), 0.0001)
    q[:, :4] = 0.24985
    q /= q.sum(axis=-1, keepdims=True)
    g_full = token_gradients("sc", p, q, y, adv)
    g_topk = token_gradients("sc_topk", p, q, y, adv, topk=4)
    np.testing.assert_allclose(g_topk, g_full, atol=1e-3)


def test_drift_norm_zero_on_policy(dists):
    p, _, _, _ = dists
    assert drift_norm(p, p) == pytest.approx(0.0, abs=1e-12)


def test_all_methods_registered():
    assert set(METHODS) == {
        "naive", "sc", "sc_topk", "tis", "mis", "sc+tis", "sc+mis", "oracle",
    }


def test_unknown_method_raises(dists):
    p, q, y, adv = dists
    with pytest.raises(ValueError):
        token_gradients("bogus", p, q, y, adv)

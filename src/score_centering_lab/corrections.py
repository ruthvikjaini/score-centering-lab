"""Correction methods for policy gradients under training-inference mismatch.

All gradients are ascent directions on the expected reward, per sampled
token, in the trainer's logit space at the token's state. ``p`` is the
trainer's next-token distribution, ``q`` the sampler's, ``y`` the sampled
token, ``A`` the (group-centred) advantage. ``r_v = p_v / q_v`` is the
token importance ratio.

The paper's identities, specialised to a softmax policy (Sections 4, A.2):

* The score is ``s_y = e_y - p`` and its expectation under the sampler is
  ``E_q[s] = q - p`` -- the *drift*. On-policy (``q == p``) drift is zero.
* Score centering (Eq. 4) replaces the score by ``s_y - E_q[s] = e_y - q``.
  Its expectation under the sampler is exactly zero at every prefix, so
  the drift term of Eq. 3 is cancelled exactly (Eq. 5), with no importance
  ratio, masking, or clipping.
* Truncated importance sampling (TIS, paper Table 1 / CISPO): weight
  ``w_v = min(r_v, 2)`` -- the ratio is clipped only at the top, never
  at the bottom. The gradient is ``w_y * s_y``.
* Masked importance sampling (MIS, paper Table 1 / IcePop): weight
  ``w_v = r_v`` inside the band ``0.5 <= r_v <= 5`` and ``0`` outside it
  (the token's gradient is dropped). The gradient is ``w_y * s_y``.
* Compositions (paper Appendix A.2, Eq. 11): IS first reweights the
  scores, then score centering is applied *on top of the reweighted
  scores* -- subtract the expectation of the *weighted* score, not the
  weighted expectation of the raw score:

      R * (w_y * s_y - E_q[w * s]),   E_q[w * s] = sum_v q_v w_v s_v.

  In logit space, ``E_q[w*s] = c - C*p`` with ``c_v = q_v*w_v`` and
  ``C = sum_v c_v``. Note the subtlety: the wrong composition
  ``w_y * (s_y - E_q[s])`` subtracts ``w_y`` times the drift -- a
  token-dependent amount -- and reintroduces drift; Eq. 11 subtracts the
  constant ``E_q[w*s]``, so the expected update under a constant reward
  is exactly zero for *any* weighting function. Sanity check (paper
  A.2): exact IS (``w = r``) gives ``E_q[w*s] = 0``, so there is nothing
  left to centre.

The top-k variant implements the paper's Section 4.1 approximation: only
the sampler's top-k probabilities are logged and the tail is modelled with
the trainer's distribution rescaled to the sampler's tail mass.
"""

from __future__ import annotations

import numpy as np

METHODS = ("naive", "sc", "sc_topk", "tis", "mis", "sc+tis", "sc+mis", "oracle")

# TIS clips the importance ratio at 2 (paper Table 1: Low 0, High 2.0).
TIS_CAP = 2.0
# MIS keeps the ratio itself inside [0.5, 5] and drops the token outside
# (paper Table 1: Low 0.5, High 5.0; JAX code in Appendix A.3).
MIS_LOW = 0.5
MIS_HIGH = 5.0


def topk_sampler_approx(p: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """Approximate the sampler distribution from its top-k probabilities.

    ``p``, ``q``: (..., K) trainer / sampler distributions. Returns ``q~``
    with the sampler's exact mass on its top-k tokens and the trainer's
    shape on the tail, rescaled so the tail carries the sampler's tail mass
    (paper Section 4.1, Eq. 8's ``rho``). ``q~`` is a valid distribution.
    """
    k = min(int(k), q.shape[-1])
    # Indices of the sampler's top-k tokens, descending.
    top_idx = np.argsort(-q, axis=-1)[..., :k]
    head_mask = np.zeros_like(q, dtype=bool)
    np.put_along_axis(head_mask, top_idx, True, axis=-1)
    q_head_mass = np.where(head_mask, q, 0.0).sum(axis=-1, keepdims=True)
    p_tail_mass = np.where(head_mask, 0.0, p).sum(axis=-1, keepdims=True)
    q_tail_mass = 1.0 - q_head_mass
    rho = np.where(p_tail_mass > 0, q_tail_mass / np.maximum(p_tail_mass, 1e-12), 0.0)
    q_tilde = np.where(head_mask, q, rho * p)
    # Renormalise against floating-point drift; the construction already
    # sums to 1 up to rounding.
    return q_tilde / q_tilde.sum(axis=-1, keepdims=True)


def _is_weights(method: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-token importance weights ``w_v`` for the IS-based methods."""
    ratio = p / np.maximum(q, 1e-12)
    if method in ("tis", "sc+tis"):
        return np.minimum(ratio, TIS_CAP)
    if method in ("mis", "sc+mis"):
        return np.where((ratio >= MIS_LOW) & (ratio <= MIS_HIGH), ratio, 0.0)
    raise ValueError(f"not an IS method: {method!r}")


def token_gradients(
    method: str,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    advantages: np.ndarray,
    topk: int = 8,
) -> np.ndarray:
    """Per-token ascent gradients in logit space, shape (..., K).

    ``p``, ``q``: (..., K) distributions; ``y``: (...,) sampled tokens;
    ``advantages``: (...,) one per token (broadcast over the vocab axis).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
    one_hot = np.zeros_like(p)
    np.put_along_axis(one_hot, y[..., None], 1.0, axis=-1)
    naive_score = one_hot - p  # s_y

    if method == "oracle":
        # Reference: no mismatch at all, so q == p and this is plain PG.
        return advantages[..., None] * naive_score

    if method in ("sc", "sc_topk"):
        q_eff = q if method == "sc" else topk_sampler_approx(p, q, topk)
        # Centred score: e_y - q (Eq. 4). Its expectation under q is
        # sum_v q_v (e_v - q) = q - q = 0, exactly.
        score = one_hot - q_eff
        return advantages[..., None] * score

    if method == "naive":
        return advantages[..., None] * naive_score

    # IS methods and their compositions with score centering.
    w = _is_weights(method, p, q)
    w_y = np.take_along_axis(w, y[..., None], axis=-1)[..., 0]
    if method in ("tis", "mis"):
        return (advantages * w_y)[..., None] * naive_score

    # sc+tis / sc+mis: Eq. 11 -- w_y * s_y minus E_q[w * s].
    # E_q[w*s] = sum_v q_v w_v (e_v - p) = c - C*p, c_v = q_v w_v.
    c = q * w
    big_c = c.sum(axis=-1, keepdims=True)
    expected_weighted_score = c - big_c * p
    composed = w_y[..., None] * naive_score - expected_weighted_score
    return advantages[..., None] * composed


def expected_score(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The drift vector ``E_q[s] = q - p`` at each state (paper Eq. 3).

    Shape (..., K). Zero iff the sampler matches the trainer.
    """
    return q - p


def drift_norm(p: np.ndarray, q: np.ndarray) -> float:
    """Mean L2 norm of the drift vector over the given states."""
    return float(np.linalg.norm(expected_score(p, q), axis=-1).mean())

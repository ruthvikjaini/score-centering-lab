"""Adam optimiser (Kingma & Ba, 2014) for the shared-parameter policy.

Plain SGD needs a different step size per parameter here: the features are
sparse one-hots, so a weight is updated by only a few tokens per step
while the bias is updated by every token. Adam's per-coordinate adaptive
step (the standard optimiser for policy-gradient training, and the one
used in the paper's LLM experiments) makes the step size independent of
how many tokens share a parameter.
"""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, shape, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> None:
        """In-place Adam ascent step (we maximise reward, so grad is the
        ascent direction)."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        params += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

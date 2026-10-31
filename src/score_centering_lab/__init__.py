"""Score-centering lab: training-inference mismatch in policy-gradient RL.

A small, exact, CPU-only testbed for Marek & Ryabinin's score centering
(arXiv:2609.20807). A tabular softmax policy generates short digit sequences
for a verifiable target-sum task; the sampler is a biased copy of the
trainer (weight noise, low-precision quantisation, or staleness), exactly
the training-inference mismatch (TIM) the paper studies. Correction methods
(naive policy gradient, score centering, truncated/masked importance
sampling, and their compositions) are compared on the same task, data, and
seeds.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

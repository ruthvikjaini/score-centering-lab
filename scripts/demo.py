"""Quick demo: train the oracle and the naive / score-centering policies
under severe weight-noise mismatch, and print the learning curves.

Usage: python scripts/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from score_centering_lab.policy import ExactSampler, WeightNoiseSampler
from score_centering_lab.train import run_training


def main() -> None:
    factory = lambda p: WeightNoiseSampler(p, sigma=0.6, rng=np.random.default_rng(7))
    rows = {}
    for method in ("oracle", "naive", "sc", "sc+tis"):
        rec = run_training(method, factory, seed=0, steps=400, lr=0.05, eval_every=50)
        rows[method] = rec
        print(f"{method:8s} final exact-match {rec['final_exact_match']:.3f} "
              f"(peak {rec['peak_exact_match']:.3f})", flush=True)
    print()
    print("eval step | " + " | ".join(f"{m:>8s}" for m in rows))
    for i, step in enumerate(rows["oracle"]["eval_steps"]):
        print(f"{step:9d} | " + " | ".join(
            f"{rows[m]['history']['eval_exact_match'][i]:8.3f}" for m in rows
        ))


if __name__ == "__main__":
    main()

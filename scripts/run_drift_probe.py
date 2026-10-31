"""Run the constant-reward drift probe for every method and write results.

Writes ``results/drift_probe.json``: the exact expected update norm at
initialisation (paper Eq. 3 vs Eq. 5) and the training-dynamics traces
(KL to init, KL to sampler) for each method under weight-noise mismatch,
plus a no-mismatch control run that establishes the random-walk noise
floor.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from score_centering_lab.drift import run_drift_probe
from score_centering_lab.policy import ExactSampler, WeightNoiseSampler

ROOT = Path(__file__).resolve().parents[1]

METHODS = ["naive", "sc", "sc_topk", "tis", "mis", "sc+tis", "sc+mis"]


def main() -> None:
    sigma = 0.6
    factory = lambda p: WeightNoiseSampler(p, sigma=sigma, rng=np.random.default_rng(7))
    results = {}
    t0 = time.time()
    for method in METHODS:
        recs = []
        for seed in (0, 1, 2):
            rec = run_drift_probe(method, factory, seed=seed, steps=150, lr=1.0)
            recs.append(rec)
        results[method] = {
            "initial_expected_update_norm_mean": float(
                np.mean([r["initial_expected_update_norm"] for r in recs])
            ),
            "final_expected_update_norm_mean": float(
                np.mean([r["final_expected_update_norm"] for r in recs])
            ),
            "final_kl_to_init_mean": float(np.mean([r["final_kl_to_init"] for r in recs])),
            "final_kl_to_sampler_mean": float(np.mean([r["final_kl_to_sampler"] for r in recs])),
            "initial_kl_to_sampler_mean": float(np.mean([r["initial_kl_to_sampler"] for r in recs])),
            "seeds": recs,
        }
        print(f"{method:8s} done ({time.time()-t0:.0f}s)", flush=True)
    # No-mismatch control: the random-walk noise floor.
    floor = [run_drift_probe("naive", lambda p: ExactSampler(p), seed=s, steps=150, lr=1.0) for s in (0, 1, 2)]
    results["control_no_mismatch"] = {
        "final_kl_to_init_mean": float(np.mean([r["final_kl_to_init"] for r in floor])),
        "final_kl_to_sampler_mean": float(np.mean([r["final_kl_to_sampler"] for r in floor])),
        "initial_expected_update_norm_mean": float(np.mean([r["initial_expected_update_norm"] for r in floor])),
        "seeds": floor,
    }
    out = ROOT / "results" / "drift_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

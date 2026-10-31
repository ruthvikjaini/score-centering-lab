"""Run the full method x mismatch experiment grid and write results.

Grid (defaults): mismatch settings x correction methods x seeds, each a
300-400 step training run on the target-sum task. Writes
``results/experiment.json`` (one record per run, with the full eval
trajectory) and ``results/measured.json`` (the aggregated headline table
the README quotes -- every number in it is computed from the run records
by ``scripts/crossfoot.py``).

Mismatch settings mirror the paper's Section 5:
  noise-<sigma>   Gaussian weight noise on the sampler (Section 5.2)
  quant-<step>    fixed-step weight quantisation of the sampler (5.3)
  stale-<k>       sampler refreshed only every k steps (Section 5.3)
  oracle          no mismatch (reference ceiling)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from score_centering_lab.corrections import METHODS
from score_centering_lab.policy import (
    ExactSampler,
    LogitLevelsSampler,
    QuantizedSampler,
    StaleSampler,
    WeightNoiseSampler,
)
from score_centering_lab.train import run_training

ROOT = Path(__file__).resolve().parents[1]


def make_factory(kind: str):
    if kind == "oracle":
        return lambda policy: ExactSampler(policy)
    if kind.startswith("noise-"):
        sigma = float(kind.split("-", 1)[1])
        return lambda policy: WeightNoiseSampler(policy, sigma, np.random.default_rng(7))
    if kind.startswith("quant-"):
        step = float(kind.split("-", 1)[1])
        return lambda policy: QuantizedSampler(policy, step=step)
    if kind.startswith("stale-"):
        k = int(kind.split("-", 1)[1])
        return lambda policy: StaleSampler(policy, sync_every=k)
    if kind.startswith("levels-"):
        n = int(kind.split("-", 1)[1])
        return lambda policy: LogitLevelsSampler(policy, levels=n)
    raise ValueError(f"unknown mismatch kind {kind!r}")


DEFAULT_SETTINGS = [
    "noise-1.0",
    "noise-1.5",
    "noise-2.0",
    "noise-3.0",
    "stale-64",
    "stale-128",
]
DEFAULT_METHODS = ["naive", "sc", "sc_topk", "tis", "mis", "sc+tis", "sc+mis", "oracle"]


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-(setting, method) means over seeds."""
    table = {}
    for rec in records:
        key = (rec["setting"], rec["method"])
        table.setdefault(key, []).append(rec)
    out = {}
    for (setting, method), runs in sorted(table.items()):
        finals = [r["final_exact_match"] for r in runs]
        peaks = [r["peak_exact_match"] for r in runs]
        drifts = [r["mean_drift_norm"] for r in runs]
        collapsed = sum(r["collapsed"] for r in runs)
        steps = [r["steps_to_080"] for r in runs if r["steps_to_080"] is not None]
        out[f"{setting} / {method}"] = {
            "setting": setting,
            "method": method,
            "n_seeds": len(runs),
            "final_exact_match_mean": float(np.mean(finals)),
            "final_exact_match_seeds": [round(f, 3) for f in finals],
            "peak_exact_match_mean": float(np.mean(peaks)),
            "mean_drift_norm": float(np.mean(drifts)),
            "collapsed_runs": collapsed,
            "steps_to_080_mean": float(np.mean(steps)) if steps else None,
            "steps_to_080_seeds": steps,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT / "results" / "experiment.json"))
    args = ap.parse_args()

    records = []
    t0 = time.time()
    for setting in args.settings:
        factory = make_factory(setting)
        for method in args.methods:
            for seed in args.seeds:
                rec = run_training(method, factory, seed=seed, steps=args.steps, lr=args.lr)
                rec["setting"] = setting
                records.append(rec)
                print(
                    f"{setting:12s} {method:8s} seed={seed} "
                    f"final={rec['final_exact_match']:.3f} peak={rec['peak_exact_match']:.3f} "
                    f"collapsed={rec['collapsed']} ({time.time()-t0:.0f}s)",
                    flush=True,
                )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    measured = aggregate(records)
    (ROOT / "results" / "measured.json").write_text(json.dumps(measured, indent=2))
    print(f"wrote {out_path} ({len(records)} runs) and results/measured.json")


if __name__ == "__main__":
    main()

"""Cross-foot the README's headline numbers against the raw run output.

Recomputes every figure quoted in README.md from ``results/experiment.json``
(the per-run records) and ``results/drift_probe.json``, and fails if any
README number disagrees. The README is generated from ``measured.json``
by hand, so this script is the check that no number was transcribed wrong
or invented: every headline value must trace to a source row.

Usage: python scripts/crossfoot.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)
    else:
        print(f"  ok: {message}")


def main() -> int:
    failures: list[str] = []
    measured = load_json(ROOT / "results" / "measured.json")
    experiment = load_json(ROOT / "results" / "experiment.json")
    drift = load_json(ROOT / "results" / "drift_probe.json")
    readme = (ROOT / "README.md").read_text()

    print("Cross-footing README figures against results/*.json ...")

    # 1. Every measured.json cell recomputes from the per-run records.
    by_key: dict[tuple[str, str], list[dict]] = {}
    for rec in experiment:
        by_key.setdefault((rec["setting"], rec["method"]), []).append(rec)
    for key, cell in measured.items():
        runs = by_key[(cell["setting"], cell["method"])]
        finals = [r["final_exact_match"] for r in runs]
        mean = sum(finals) / len(finals)
        check(
            abs(mean - cell["final_exact_match_mean"]) < 1e-9,
            f"measured {key} mean {cell['final_exact_match_mean']:.4f} == recomputed {mean:.4f}",
            failures,
        )
        check(
            cell["n_seeds"] == len(runs) and cell["n_seeds"] == len(cell["final_exact_match_seeds"]),
            f"measured {key} seed count {cell['n_seeds']} matches records",
            failures,
        )
        check(
            cell["collapsed_runs"] == sum(r["collapsed"] for r in runs),
            f"measured {key} collapsed runs {cell['collapsed_runs']} matches records",
            failures,
        )

    # 2. Every number quoted in the README's results table appears in
    # measured.json (to 3 decimals) -- no invented figures.
    numbers_in_readme = sorted(set(re.findall(r"\b0\.\d{3}\b", readme)))
    numbers_in_measured = set()
    for cell in measured.values():
        numbers_in_measured.add(f"{cell['final_exact_match_mean']:.3f}")
        numbers_in_measured.add(f"{cell['peak_exact_match_mean']:.3f}")
        for f in cell["final_exact_match_seeds"]:
            numbers_in_measured.add(f"{f:.3f}")
    for n in numbers_in_readme:
        # Only check numbers inside the results table section.
        table = readme.split("## Results")[1].split("## ")[0] if "## Results" in readme else ""
        if n in table:
            check(n in numbers_in_measured, f"README number {n} traces to measured.json", failures)

    # 3. Drift-probe numbers in the README trace to drift_probe.json.
    drift_numbers = set()
    for method, cell in drift.items():
        for k, v in cell.items():
            if isinstance(v, float):
                drift_numbers.add(f"{v:.3f}")
                drift_numbers.add(f"{v:.4f}")
    drift_section = readme.split("## Drift probe")[1].split("## ")[0] if "## Drift probe" in readme else ""
    for n in sorted(set(re.findall(r"\b0\.\d{3,4}\b", drift_section))):
        check(n in drift_numbers, f"README drift number {n} traces to drift_probe.json", failures)

    # 4. Sanity: the grid is complete -- every setting x method has 5 seeds.
    for (setting, method), runs in sorted(by_key.items()):
        check(len(runs) == 5, f"grid complete: {setting}/{method} has 5 seeds", failures)

    if failures:
        print(f"\nFAILED: {len(failures)} cross-foot check(s)")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("\nAll cross-foot checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

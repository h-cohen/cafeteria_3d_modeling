"""Compare two runs' scorecards metric by metric.

    python -m muontomo.compare runs/exp01 runs/exp02 [--json] [--fail-on-regression]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import REGISTRY, flatten

HEADLINE = ("fidelity.chi2_ndof", "crossval.cv_pearson", "volume.z_eff_width_m", "truth.ssim3d")


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict:
    a = flatten(json.loads((Path(run_a) / "metrics.json").read_text()))
    b = flatten(json.loads((Path(run_b) / "metrics.json").read_text()))
    rows = []
    for key, spec in REGISTRY.items():
        if key not in a or key not in b:
            continue
        va, vb = a[key], b[key]
        delta = vb - va
        band = spec["band"] * max(abs(va), 1e-9) if abs(va) > 1 else spec["band"]
        if abs(delta) <= band:
            verdict = "same"
        else:
            improved = (delta > 0) == spec["higher_is_better"]
            verdict = "IMPROVED" if improved else "REGRESSED"
        rows.append({"metric": key, "a": va, "b": vb, "delta": delta, "verdict": verdict})
    n_imp = sum(r["verdict"] == "IMPROVED" for r in rows)
    n_reg = sum(r["verdict"] == "REGRESSED" for r in rows)
    headline_reg = any(r["verdict"] == "REGRESSED" and r["metric"] in HEADLINE for r in rows)
    overall = "REGRESSED" if headline_reg else ("IMPROVED" if n_imp > n_reg else ("REGRESSED" if n_reg > n_imp else "SAME"))
    return {"rows": rows, "improved": n_imp, "regressed": n_reg, "verdict": overall,
            "run_a": str(run_a), "run_b": str(run_b)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on-regression", action="store_true")
    args = ap.parse_args(argv)
    res = compare_runs(args.run_a, args.run_b)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        w = max(len(r["metric"]) for r in res["rows"]) if res["rows"] else 20
        print(f"{'metric':<{w}} {'A':>10} {'B':>10} {'delta':>10}  verdict")
        for r in res["rows"]:
            print(f"{r['metric']:<{w}} {r['a']:>10.4g} {r['b']:>10.4g} {r['delta']:>+10.4g}  {r['verdict']}")
        print(f"\nVERDICT: {res['verdict']} ({res['improved']} better, {res['regressed']} worse)")
    if args.fail_on_regression and res["verdict"] == "REGRESSED":
        sys.exit(1)


if __name__ == "__main__":
    main()

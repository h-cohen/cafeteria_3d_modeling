"""CLI: run one or all enhancement techniques on a run and write outputs.

    python -m muontomo.enhance --run runs/production --method guided|dip|pnp|all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .base import REGISTRY
from .context import load_context
from .report import render_method_png, render_report
from .verify import verify


def run_method(ctx, method: str) -> dict:
    enhancer = REGISTRY[method]
    layer = enhancer.enhance(ctx)
    extra = getattr(enhancer, "last_info", {}) or {}
    metrics = verify(ctx, layer, runtime_s=extra.get("runtime_s"), extra=extra)
    return {"layer": np.asarray(layer, np.float32), "metrics": metrics}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", default="all", choices=["guided", "dip", "pnp", "all"])
    args = ap.parse_args(argv)

    ctx = load_context(args.run)
    methods = ["guided", "dip", "pnp"] if args.method == "all" else [args.method]
    out_dir = ctx.run / "enhance"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ctx.run / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in methods:
        print(f"[{m}] running ...", flush=True)
        try:
            res = run_method(ctx, m)
        except Exception as e:
            print(f"[{m}] FAILED: {e}", flush=True)
            continue
        results[m] = res
        np.save(out_dir / f"{m}.npy", res["layer"])
        (out_dir / f"{m}_metrics.json").write_text(json.dumps(res["metrics"], indent=2) + "\n")
        render_method_png(ctx, m, res["layer"], img_dir / f"enhance_{m}.png")
        print(f"[{m}] {res['metrics']}", flush=True)

    if len(results) > 1:
        report = render_report(ctx, results, out_dir)
        print(f"report -> {report}", flush=True)


if __name__ == "__main__":
    main()

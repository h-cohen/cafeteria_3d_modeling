"""Volume export contract for the 3D viewer: volume.npy + meta.json.

meta.json carries everything the viewer needs to place, scale, and label the
volume without re-deriving anything from the run's config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import VoxelGrid


def export_volume(run_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    with np.load(run / "volume.npz") as z:
        rho = z["rho"].astype(np.float32)
        origin = tuple(float(v) for v in z["origin"])
        spacing = float(z["spacing"])

    metrics = {}
    mfile = run / "metrics.json"
    if mfile.exists():
        metrics = json.loads(mfile.read_text()).get("headline", {})

    pos = np.maximum(rho, 0.0)
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0

    np.save(out / "volume.npy", rho)
    meta = {
        "shape": list(rho.shape),
        "axis_order": "xyz",
        "origin_m": list(origin),
        "spacing_m": spacing,
        "units": "relative opacity density [1/m]",
        "value_range": [float(rho.min()), float(rho.max())],
        "suggested_iso": [round(0.3 * p99, 6), round(0.6 * p99, 6)],
        "run": run.name,
        "headline_metrics": metrics,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out / "volume.npy"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    p = export_volume(args.run, args.out)
    print(f"exported {p} (+ meta.json)")


if __name__ == "__main__":
    main()

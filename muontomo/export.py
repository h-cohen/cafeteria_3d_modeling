"""Volume export contract for the 3D viewer: volume.npy + meta.json.

meta.json carries everything the viewer needs to place, scale, and label the
volume without re-deriving anything from the run's config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import RunConfig
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

    detectors = []
    viewer_crop_xy_m = None
    cfile = run / "config.json"
    if cfile.exists():
        geom = RunConfig.load(cfile).geometry
        if geom.viewer_crop_xy_m is not None:
            viewer_crop_xy_m = [list(geom.viewer_crop_xy_m[0]), list(geom.viewer_crop_xy_m[1])]
        for pid, p in geom.poses.items():
            pose = geom.pose(pid)
            detectors.append({
                "id": pid, "x": pose.x, "y": pose.y, "z": pose.z, "yaw_deg": pose.yaw_deg,
                "aperture_m": geom.aperture_m, "height_m": geom.detector_height_m,
            })

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
        "detectors": detectors,
        "viewer_crop_xy_m": viewer_crop_xy_m,
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

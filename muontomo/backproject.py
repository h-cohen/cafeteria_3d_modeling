"""Single-plane backprojection of the measured opacity, -ln(T), per detector.

This is the model-free view of the data: each angular bin of a detector's
calibrated transmission map is projected along its ray onto a horizontal plane
at the ceiling height. No solver, no regularization -- what you see is what the
detector measured, just placed in room coordinates. It is the ground-truth
anchor the 3D reconstruction should be compared against.

    python -m muontomo.backproject --run runs/production
"""

from __future__ import annotations

import numpy as np

from .config import GeometryConfig
from .geometry import pose_rotation


def backproject_opacity(
    tmaps: dict,
    geom: GeometryConfig,
    z_m: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Project each pose's measured opacity onto the plane z=z_m.

    xs, ys are the world-coordinate pixel centers of the output grid.
    Returns ({pose_id: grid}, mean_grid); grids are [len(xs), len(ys)], NaN
    where a pose has no (masked-out) coverage. Rays start at (pose.x, pose.y,
    pose.z), matching the forward model's convention (see raycast._pose_block).
    """
    from scipy.interpolate import griddata

    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    per_pose = {}
    for pid, tm in tmaps.items():
        pose = geom.pose(pid)
        lever = z_m - pose.z
        tx, ty = np.meshgrid(tm.txcenters, tm.tycenters, indexing="ij")
        R = pose_rotation(pose)
        px = pose.x + (R[0, 0] * tx + R[0, 1] * ty) * lever
        py = pose.y + (R[1, 0] * tx + R[1, 1] * ty) * lever
        with np.errstate(divide="ignore"):
            op = -np.log(np.clip(tm.T, 0.05, None))
        ok = tm.mask & np.isfinite(op)
        if not ok.any():
            per_pose[pid] = np.full(GX.shape, np.nan)
            continue
        per_pose[pid] = griddata(
            (px[ok], py[ok]), np.clip(op[ok], 0, None), (GX, GY), method="linear"
        )
    stack = np.stack(list(per_pose.values()))
    n_ok = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(n_ok > 0, np.nansum(stack, axis=0) / np.maximum(n_ok, 1), np.nan)
    return per_pose, mean


def run_backprojection(run_dir, z_m: float | None = None, res_m: float = 0.08):
    """Backproject a run's measured data at its fitted layer height.

    Returns (xs, ys, z_m, per_pose, mean) on the run's grid_xy_m extent.
    """
    import json
    from pathlib import Path

    from .calibration import transmission_maps
    from .config import RunConfig
    from .io import load_dataset

    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    if z_m is None:
        if cfg.reconstruction.layered_zs:
            z_m = float(cfg.reconstruction.layered_zs[0])
        else:  # fall back to the reconstruction's peak-mass height
            meta = json.loads((run / "meta.json").read_text())
            rho = np.load(run / "volume.npy")
            zprof = np.maximum(rho, 0).sum(axis=(0, 1))
            z_m = meta["origin_m"][2] + int(np.argmax(zprof)) * meta["spacing_m"]
    ds = load_dataset(cfg.data)
    tmaps = transmission_maps(ds, cfg.binning, cfg.calibration)
    (x0, x1), (y0, y1) = cfg.geometry.grid_xy_m
    xs = np.arange(x0 + res_m / 2, x1, res_m)
    ys = np.arange(y0 + res_m / 2, y1, res_m)
    per_pose, mean = backproject_opacity(tmaps, cfg.geometry, z_m, xs, ys)
    return xs, ys, z_m, per_pose, mean


def render_backprojection_png(run_dir) -> None:
    """Write images/backproject.png: per-pose panels + the mean, at layer height."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, z_m, per_pose, mean = run_backprojection(run_dir)
    panels = list(per_pose.items()) + [("mean of poses", mean)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), constrained_layout=True)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    vmax = np.nanpercentile(mean, 99)
    for ax, (name, img) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(img.T, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vmax)
        ax.set(title=f"{name}: measured opacity at z={z_m:.2f} m", xlabel="x (m)", ylabel="y (m)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    img_dir = Path(run_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "backproject.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    render_backprojection_png(args.run)
    print(f"{args.run}/images/backproject.png")


if __name__ == "__main__":
    main()

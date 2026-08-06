"""Beam-level, model-free verification of a reconstruction.

The ceiling structure is a set of beams, and the beams are visible directly in
each detector's calibrated transmission map -- so the reconstruction can be
verified at the level of the physics claim, without trusting the solver:

1. Opacity profiles: collapse each pose's -ln(T) map along the beam direction
   (vertical beams -> profile vs tan_x; horizontal -> profile vs tan_y).
2. Parallax height: project both poses' profiles onto a candidate plane z and
   correlate them. Real structure at height z* makes the profiles align at z*
   (the two detectors are separated by a known baseline, so the alignment
   height is a triangulation, independent of any reconstruction). The x and y
   baselines give two independent estimates.
3. Beam match: peak positions in the data profiles (at the parallax height)
   vs peak positions in the reconstruction's layer slice, beam by beam.

    python -m muontomo.beams --run runs/production
"""

from __future__ import annotations

import numpy as np

CLIP_T = 0.05  # transmission floor before -ln(T)


def _opacity(tm) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(tm.mask, -np.log(np.clip(tm.T, CLIP_T, None)), np.nan)


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmean without the all-NaN RuntimeWarning (all-NaN rows -> NaN)."""
    n = np.isfinite(a).sum(axis=axis)
    s = np.nansum(a, axis=axis)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def profile(tm, axis: str = "x", band: float = 0.32) -> tuple[np.ndarray, np.ndarray]:
    """Opacity profile vs tan(theta_axis), averaged over |t_other| < band."""
    op = _opacity(tm)
    if axis == "x":
        sel = np.abs(tm.tycenters) < band
        return tm.txcenters, _nanmean(op[:, sel], axis=1)
    sel = np.abs(tm.txcenters) < band
    return tm.tycenters, _nanmean(op[sel, :], axis=0)


def _world_profile(t, prof, pose, axis: str, z: float, grid: np.ndarray) -> np.ndarray:
    p0 = pose.x if axis == "x" else pose.y
    w = p0 + t * (z - pose.z)
    ok = np.isfinite(prof)
    return np.interp(grid, w[ok], prof[ok], left=np.nan, right=np.nan)


def parallax_scan(
    tmaps: dict, geom, axis: str, grid: np.ndarray, zs: np.ndarray, band: float = 0.32
) -> tuple[np.ndarray, float, float]:
    """Correlation of the two poses' world-projected profiles vs assumed height.

    Returns (correlations, best_z, best_corr). Only meaningful with >= 2 poses.
    """
    profs = {pid: profile(tm, axis, band) for pid, tm in tmaps.items()}
    pids = list(profs)
    corrs = np.full(len(zs), np.nan)
    for k, z in enumerate(zs):
        ws = [
            _world_profile(*profs[pid], geom.pose(pid), axis, float(z), grid)
            for pid in pids
        ]
        ok = np.all([np.isfinite(w) for w in ws], axis=0)
        if ok.sum() < 10:
            continue
        a, b = ws[0][ok], ws[1][ok]
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        if denom > 0:
            corrs[k] = float((a * b).sum() / denom)
    i = int(np.nanargmax(corrs))
    return corrs, float(zs[i]), float(corrs[i])


def beam_peaks(grid: np.ndarray, prof: np.ndarray, prom_sigmas: float = 0.5) -> np.ndarray:
    from scipy.signal import find_peaks

    ok = np.isfinite(prof)
    if ok.sum() < 5:
        return np.array([])
    prom = prom_sigmas * float(np.nanstd(prof))
    idx, _ = find_peaks(prof[ok], prominence=prom)
    return grid[ok][idx]


def verify_beams(run_dir) -> dict:
    """Full beam-level verification of a run; also writes images/beam_verify.png."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    from .calibration import transmission_maps
    from .config import RunConfig
    from .io import load_dataset

    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    geom = cfg.geometry

    meta_grid = cfg.geometry.grid_xy_m
    xgrid = np.linspace(meta_grid[0][0] + 1.0, meta_grid[0][1] - 1.0, 400)
    ygrid = np.linspace(meta_grid[1][0] + 0.6, meta_grid[1][1] - 0.6, 400)
    zs = np.linspace(cfg.geometry.grid_z_m[0], cfg.geometry.grid_z_m[1], 81)

    cx, zx, rx = parallax_scan(tmaps, geom, "x", xgrid, zs)
    cy, zy, ry = parallax_scan(tmaps, geom, "y", ygrid, zs, band=0.45)

    # data beam positions at the x-parallax height (mean of the poses' profiles)
    profs = {pid: profile(tm, "x") for pid, tm in tmaps.items()}
    world = [
        _world_profile(*profs[pid], geom.pose(pid), "x", zx, xgrid) for pid in profs
    ]
    data_prof = _nanmean(np.stack(world), axis=0)
    pk_data = beam_peaks(xgrid, data_prof)

    # horizontal beams: same, along y at the y-parallax height
    profs_y = {pid: profile(tm, "y", band=0.45) for pid, tm in tmaps.items()}
    world_y = [
        _world_profile(*profs_y[pid], geom.pose(pid), "y", zy, ygrid) for pid in profs_y
    ]
    pk_data_y = beam_peaks(ygrid, _nanmean(np.stack(world_y), axis=0))

    # reconstruction beam positions in its layer slice (volume.npz is the
    # canonical solver output; volume.npy is a derived viewer export that may
    # be stale)
    with np.load(run / "volume.npz") as vz:
        rho = vz["rho"].astype(np.float64)
        origin = [float(v) for v in vz["origin"]]
        sp = float(vz["spacing"])
    pos = np.maximum(rho, 0.0)
    iz = int(np.argmax(pos.sum(axis=(0, 1))))
    z_recon = origin[2] + iz * sp
    sl = pos[:, :, iz]
    ys_r = origin[1] + (np.arange(sl.shape[1]) + 0.5) * sp
    xs_r = origin[0] + (np.arange(sl.shape[0]) + 0.5) * sp
    yband = (ys_r > -2.0) & (ys_r < 2.0)
    prof_r = gaussian_filter1d(sl[:, yband].mean(axis=1), max(0.12 / sp, 1e-9))
    pk_recon = beam_peaks(xgrid, np.interp(xgrid, xs_r, prof_r))

    offsets = [
        float(pk_recon[np.argmin(np.abs(pk_recon - p))] - p) for p in pk_data
    ] if len(pk_recon) else []
    report = {
        "parallax_z_x_m": round(zx, 2),
        "parallax_corr_x": round(rx, 3),
        "parallax_z_y_m": round(zy, 2),
        "parallax_corr_y": round(ry, 3),
        "recon_layer_z_m": round(z_recon, 2),
        "n_beams_data": int(len(pk_data)),
        "n_beams_recon": int(len(pk_recon)),
        "beams_y_data_m": [round(float(p), 2) for p in pk_data_y],
        "beams_x_data_m": [round(float(p), 2) for p in pk_data],
        "beams_x_recon_m": [round(float(p), 2) for p in pk_recon],
        "beam_offsets_m": [round(o, 2) for o in offsets],
        "mean_abs_beam_offset_m": round(float(np.mean(np.abs(offsets))), 3) if offsets else None,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(zs, cx, label=f"x-baseline (vertical beams): best z={zx:.2f}, corr={rx:.2f}")
    ax.plot(zs, cy, label=f"y-baseline (horizontal beam): best z={zy:.2f}, corr={ry:.2f}")
    ax.axvline(z_recon, color="k", ls="--", lw=1, label=f"reconstruction layer z={z_recon:.2f}")
    ax.set(xlabel="assumed plane height z (m)", ylabel="pose-pose profile correlation",
           title="model-free height triangulation by beam parallax")
    ax.legend()
    ax = axes[1]
    for pid, w in zip(profs, world):
        ax.plot(xgrid, w, alpha=0.6, label=f"{pid} opacity @ z={zx:.2f}")
    pr = np.interp(xgrid, xs_r, prof_r)
    sc = np.nanstd(data_prof) / max(np.nanstd(pr), 1e-12)
    ax.plot(xgrid, (pr - np.nanmean(pr)) * sc + np.nanmean(data_prof), "k", lw=2,
            label="reconstruction slice profile (scaled)")
    for p in pk_data:
        ax.axvline(p, color="gray", ls=":", lw=1)
    ax.set(xlabel="x (m)", ylabel="opacity -ln(T)",
           title=f"beam positions: raw data (dotted) vs reconstruction "
                 f"(mean |offset| = {report['mean_abs_beam_offset_m']} m)")
    ax.legend(fontsize=9)
    img_dir = run / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "beam_verify.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    report = verify_beams(args.run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

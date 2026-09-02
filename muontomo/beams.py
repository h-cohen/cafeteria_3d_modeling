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


def beam_peaks_subbin(grid: np.ndarray, prof: np.ndarray, prom_sigmas: float = 0.5) -> np.ndarray:
    """`beam_peaks` refined to sub-bin precision by a parabola through each maximum.

    Triangulation is acutely sensitive to peak position: with baseline 1.78 m and a
    0.04 tan-unit bin, one bin of quantization in (t_0 - t_1) moves the closed-form
    height by ~1 m. Bin-centre peaks are therefore not good enough, and the parabolic
    refinement (same trick as uncertainty._focus_z) removes most of that error.
    """
    from scipy.signal import find_peaks

    ok = np.isfinite(prof)
    if ok.sum() < 5:
        return np.array([])
    g, p = grid[ok], prof[ok]
    idx, _ = find_peaks(p, prominence=prom_sigmas * float(np.nanstd(prof)))
    out = []
    for i in idx:
        if 0 < i < len(p) - 1:
            denom = p[i - 1] - 2 * p[i] + p[i + 1]
            # denom < 0 at a maximum; guard against a flat or pathological triple
            delta = 0.5 * (p[i - 1] - p[i + 1]) / denom if denom < 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            step = g[i + 1] - g[i]
            out.append(g[i] + delta * step)
        else:
            out.append(g[i])
    return np.asarray(out)


def _match_peaks(peaks: dict, poses: dict, axis: str, z0: float, tol_m: float) -> list[dict]:
    """Group per-pose tan-space peaks into features by their world position at z0.

    Each pose sees the same beam at a different angle; projected to the plane z0 they
    land on the same world coordinate (that is what makes z0 the right height). So
    project, then match nearest-neighbour across poses within `tol_m`. Returns one
    entry per feature seen by >= 2 poses: {"world0": x, "obs": {pid: t}}.
    """
    pids = list(peaks)
    ref = pids[0]
    p0 = poses[ref]
    base = getattr(p0, axis) + np.asarray(peaks[ref]) * (z0 - p0.z)
    feats = [{"world0": float(w), "obs": {ref: float(t)}} for w, t in zip(base, peaks[ref])]
    for pid in pids[1:]:
        pose = poses[pid]
        for t in peaks[pid]:
            w = getattr(pose, axis) + t * (z0 - pose.z)
            if not feats:
                continue
            d = [abs(w - f["world0"]) for f in feats]
            i = int(np.argmin(d))
            if d[i] < tol_m and pid not in feats[i]["obs"]:
                feats[i]["obs"][pid] = float(t)
    return [f for f in feats if len(f["obs"]) >= 2]


def triangulate(tmaps: dict, geom, z0: float, sigma_t: float, tol_m: float = 0.6) -> dict:
    """Joint least-squares ray intersection over every matched beam, both axes.

    Each (pose, beam) peak fixes a ray; a beam at height z and world position X is
    seen by pose i at tan angle t = (X - p_i)/(z - z_i). Imposing that ALL beams share
    one ceiling height turns the per-pair closed form z = dx/(t_0 - t_1) into an
    overdetermined fit for (z, {X_k}, {Y_m}), which is both more precise and testable:
    the residual scatter says whether a single plane actually explains the data.

    Independent of the reconstruction AND of the cross-validation autofocus -- it uses
    only peak positions in the calibrated transmission maps -- so it is a genuine
    cross-check rather than a restatement. Returns z, its standard error, and the fit
    diagnostics. `sigma_t` is the per-peak angular uncertainty (tan units).
    """
    from scipy import optimize

    poses = {pid: geom.pose(pid) for pid in tmaps}
    feats = {}
    for axis, band in (("x", 0.32), ("y", 0.45)):
        peaks = {
            pid: beam_peaks_subbin(*profile(tm, axis, band)) for pid, tm in tmaps.items()
        }
        feats[axis] = _match_peaks(peaks, poses, axis, z0, tol_m)
    nx, ny = len(feats["x"]), len(feats["y"])
    n_obs = sum(len(f["obs"]) for a in feats for f in feats[a])
    n_par = 1 + nx + ny
    if nx + ny < 1 or n_obs <= n_par:
        return {"ok": False, "reason": f"underdetermined: {n_obs} obs, {n_par} params"}

    def unpack(p):
        return p[0], p[1 : 1 + nx], p[1 + nx :]

    def resid(p):
        z, xs, ys = unpack(p)
        out = []
        for axis, vals in (("x", xs), ("y", ys)):
            for f, w in zip(feats[axis], vals):
                for pid, t in f["obs"].items():
                    pose = poses[pid]
                    out.append(((w - getattr(pose, axis)) / (z - pose.z) - t) / sigma_t)
        return np.asarray(out)

    p0 = np.concatenate([[z0], [f["world0"] for f in feats["x"]],
                         [f["world0"] for f in feats["y"]]])
    fit = optimize.least_squares(resid, p0, method="lm")
    z, xs, ys = unpack(fit.x)

    dof = n_obs - n_par
    ssr = float(np.sum(fit.fun**2))
    # Scale the covariance by the achieved residual variance rather than trusting
    # sigma_t: it absorbs a mis-set sigma_t and reports the fit's own consistency.
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * (ssr / dof)
        sigma_z = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma_z = float("nan")
    return {
        "ok": True,
        "z_m": round(float(z), 3),
        "z_sigma_m": round(sigma_z, 3),
        "n_features_x": nx,
        "n_features_y": ny,
        "n_observations": n_obs,
        "dof": dof,
        "chi2_per_dof": round(ssr / dof, 3),
        "resid_rms_tan": round(float(np.sqrt(ssr / n_obs)) * sigma_t, 5),
        "sigma_t_assumed": round(sigma_t, 5),
        "beams_x_m": [round(float(v), 3) for v in xs],
        "beams_y_m": [round(float(v), 3) for v in ys],
    }


def angular_beam_period(tm, axis: str = "x", t_window: float = 0.6) -> float:
    """Dominant angular period of the beam pattern in one detector, in tan-units.

    Measured by the FFT peak of the detrended opacity profile with parabolic sub-bin
    refinement. This is a per-detector quantity: it needs no baseline, no pose and no
    height, which is what makes it useful for closing the scale (see scale_closure).
    """
    t, p = profile(tm, axis)
    sel = np.isfinite(p) & (np.abs(t) < t_window)
    tt, pp = t[sel], p[sel]
    if len(tt) < 16:
        return float("nan")
    pp = pp - np.polyval(np.polyfit(tt, pp, 3), tt)
    freq = np.fft.rfftfreq(len(tt), tt[1] - tt[0])
    amp = np.abs(np.fft.rfft(pp * np.hanning(len(pp))))
    k = int(np.argmax(amp[2:])) + 2
    if k + 1 < len(amp):
        denom = amp[k - 1] - 2 * amp[k] + amp[k + 1]
        dk = 0.5 * (amp[k - 1] - amp[k + 1]) / denom if denom < 0 else 0.0
        f_pk = freq[k] + float(np.clip(dk, -0.5, 0.5)) * (freq[1] - freq[0])
    else:
        f_pk = freq[k]
    return float(1.0 / f_pk) if f_pk > 0 else float("nan")


def scale_closure(tmaps, geom, z_m: float) -> dict:
    """Close the (baseline, ceiling height, beam pitch) scale triangle.

    Triangulation fixes only a RATIO: z = d / (t_1 - t_2), so the height scales with the
    assumed baseline d and cannot be got from the angles alone. But each detector also
    measures the beam pattern's angular period on its own -- no baseline, pose or height
    involved -- and the physical pitch is pitch = z * period. The three quantities are
    therefore locked together: fixing ANY ONE of baseline, height or pitch by an
    independent measurement (a tape measure on the detector separation, or on the ceiling
    beam spacing) determines the other two. Reporting the triple makes the assumption
    visible instead of leaving it buried in the pose config.
    """
    p = {pid: geom.pose(pid) for pid in tmaps}
    pids = list(p)
    d = (float(np.hypot(p[pids[1]].x - p[pids[0]].x, p[pids[1]].y - p[pids[0]].y))
         if len(pids) >= 2 else float("nan"))
    per = {pid: angular_beam_period(tm) for pid, tm in tmaps.items()}
    vals = [v for v in per.values() if np.isfinite(v)]
    period = float(np.mean(vals)) if vals else float("nan")
    return {
        "baseline_m": round(d, 4),
        "angular_beam_period_tan": round(period, 5),
        "angular_period_per_pose": {k: round(v, 5) for k, v in per.items()},
        "triangulated_z_m": round(z_m, 3),
        "implied_beam_pitch_m": round(period * z_m, 3),
        # z and pitch both scale linearly with the assumed baseline; these let a reader
        # rescale to any surveyed d without re-running anything.
        "z_per_baseline": round(z_m / d, 4) if d else None,
        "pitch_per_baseline": round(period * z_m / d, 4) if d else None,
    }


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

    first = next(iter(tmaps.values()))
    bin_t = float(first.txedges[1] - first.txedges[0])
    report["triangulation"] = triangulate(
        tmaps, geom, zx, sigma_t=bin_t / np.sqrt(12.0)
    )
    tri = report["triangulation"]
    report["scale_closure"] = scale_closure(
        tmaps, geom, tri["z_m"] if tri.get("ok") else zx
    )

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

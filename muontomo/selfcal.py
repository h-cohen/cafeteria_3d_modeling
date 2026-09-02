"""Geometry self-calibration: refine the approximately-known detector poses.

Stage A (real-data, diagnostic): NCC registration between pos0/pos1 XY
back-projection maps at each canned height, plus a yaw scan. On this campaign
the parallax shift (baseline/height, order ~0.5 tan-units for a ~1.75 m
baseline at a ~3 m ceiling) is comparable to the whole angular acceptance, so
the overlap after shifting is small and the NCC curve is flat and noisy (see
runs/selfcal*/selfcal_report.json) -- image registration alone cannot resolve
it here. Stage A's numbers are recorded for inspection but are NOT used to
seed Stage B; only the user-supplied geometry prior (config.py: ~1.75 m
horizontal offset, same floor) is trustworthy enough to seed a local search.

Stage B (dataset-agnostic, authoritative): jointly refine pose offsets by
minimizing the symmetric cross-position chi2 (reconstruct from one position on
a coarse grid, score against the other, using the full aperture-aware forward
model) around the config's prior. Validated on phantom.py data with an
injected pose error (tests/test_selfcal.py); on real data it is bounded to
+-bounds_m / +-bounds_deg around the prior so it refines rather than wanders.

Usage:
    python -m muontomo.selfcal --config configs/production.json --out runs/selfcal01
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.ndimage import rotate as ndi_rotate

from .calibration import compute_transmission, prepare_angular_hist, transmission_maps
from .config import CalibrationConfig, GeometryConfig, RunConfig
from .forward import build_forward_model
from .io import Dataset, load_dataset, load_root_hist2d
from .opacity import opacity_map
from .reconstruct import fit_data, solve

XY_HEIGHTS_M = {"XY01m": 1.0, "XY02m": 2.0, "XY05m": 5.0, "XY07m": 7.0, "XY10m": 10.0}


@dataclasses.dataclass
class GeometryEstimate:
    dx: float
    dy: float
    yaw_deg: float
    ceiling_z_m: float | None
    confidence: dict


# ---------------------------------------------------------------------------
# Stage A: focus + NCC registration on the canned XY maps (real ROOT data only)


def _xy_transmission(ds: Dataset, source: str, name: str, cal: CalibrationConfig):
    cafe = load_root_hist2d(ds.sources[source], name)
    sky = load_root_hist2d(ds.sources["sky"], name)
    return compute_transmission(cafe, sky, cal, pose_id=source)


def depth_from_parallax(ds: Dataset, cal: CalibrationConfig, max_shift_bins: int = 15) -> dict:
    """Select the ceiling height by cross-position registration strength.

    NOTE (corrected): an earlier version of this docstring claimed the XY maps are
    a pure coordinate rescaling of tan-space and therefore carry no depth
    information on their own. That is wrong. The DAQ builds them by the
    single-detector shear

        t_corr = t + b/H       (b = bottom-layer hit position, H = assumed height)

    which is the intra-detector analogue of two-detector parallax focusing: it
    collapses each track to a common vertex, and is exact only at the true H.
    Unlike a rescaling, a shear of the (position, angle) light field is NOT
    invertible once position is marginalized away, so it does carry depth
    information -- weakly, since the lever is the ~0.65 m aperture rather than the
    1.92 m stereo baseline.

    Verified on the campaign data: under the shear, E[t_corr]*H - E[t_inf]*H =
    E[b], independent of H. Measured E[b] = 0.1933 +/- 0.0002 m across
    H = 1,2,5,7,10 m, in all three files and both axes; the maps are in tan units
    on txty's axis, preserve total counts exactly, and converge onto txty as
    H -> infinity. See reports/refocusing_response.md.

    This function nonetheless still scores by CROSS-position registration, which
    is the right choice: the single-detector sharpness route carries a monotonic
    lever-arm bias (back-projection fans widen with H), whereas parallax between
    the two positions cancels under a single rigid translation only at the true
    structure height. So the height with the highest best-shift NCC between pos0
    and pos1's XY-transmission maps is the depth estimate.
    """
    curve = {}
    for name, h in XY_HEIGHTS_M.items():
        reg = register_offset(ds, name, cal, max_shift_bins=max_shift_bins)
        curve[name] = {"height_m": h, **reg}
    best = max(curve, key=lambda k: curve[k]["ncc"])
    return {"curve": curve, "best_height_m": curve[best]["height_m"], "best_name": best}


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.sum(am**2) * np.sum(bm**2))
    return float(np.sum(am * bm) / denom) if denom > 0 else 0.0


def register_offset(ds: Dataset, name: str, cal: CalibrationConfig, max_shift_bins: int = 15) -> dict:
    """NCC-registration of pos1's XY-transmission map onto pos0's, over an
    integer-bin shift grid; the argmax shift is the position offset in meters."""
    t0 = _xy_transmission(ds, "pos0", name, cal)
    t1 = _xy_transmission(ds, "pos1", name, cal)
    a = np.nan_to_num(np.where(t0.mask, t0.T, 1.0), nan=1.0)
    b = np.nan_to_num(np.where(t1.mask, t1.T, 1.0), nan=1.0)
    bin_m = float(t0.txedges[1] - t0.txedges[0])
    best = (-np.inf, 0, 0)
    curve = []
    for di in range(-max_shift_bins, max_shift_bins + 1):
        for dj in range(-max_shift_bins, max_shift_bins + 1):
            score = _ncc(a, np.roll(b, (di, dj), axis=(0, 1)))
            curve.append((di, dj, score))
            if score > best[0]:
                best = (score, di, dj)
    return {
        "ncc": best[0], "shift_bins": (best[1], best[2]),
        "dx_m": best[1] * bin_m, "dy_m": best[2] * bin_m,
        "bin_m": bin_m,
    }


def register_yaw(ds: Dataset, name: str, cal: CalibrationConfig, max_deg: float = 10.0, step_deg: float = 0.5) -> dict:
    """1D NCC scan of pos1's rotation relative to pos0 at the best-focus height."""
    t0 = _xy_transmission(ds, "pos0", name, cal)
    t1 = _xy_transmission(ds, "pos1", name, cal)
    a = np.nan_to_num(np.where(t0.mask, t0.T, 1.0), nan=1.0)
    b0 = np.nan_to_num(np.where(t1.mask, t1.T, 1.0), nan=1.0)
    angles = np.arange(-max_deg, max_deg + step_deg, step_deg)
    scores = [_ncc(a, ndi_rotate(b0, ang, reshape=False, order=1, mode="nearest")) for ang in angles]
    best_i = int(np.argmax(scores))
    return {"yaw_deg": float(angles[best_i]), "ncc": float(scores[best_i]),
            "curve": list(zip(angles.tolist(), scores))}


def stage_a(cfg: RunConfig) -> GeometryEstimate:
    """Cross-position depth + offset + yaw from real-data XY maps. No 'phantom' key in cfg.data."""
    if "phantom" in cfg.data:
        raise ValueError("Stage A needs real ROOT XY back-projection maps, not a phantom dataset")
    ds = load_dataset(cfg.data)
    depth = depth_from_parallax(ds, cfg.calibration)
    best_name = depth["best_name"]
    offset = depth["curve"][best_name]
    yaw = register_yaw(ds, best_name, cfg.calibration)
    return GeometryEstimate(
        dx=offset["dx_m"], dy=offset["dy_m"], yaw_deg=yaw["yaw_deg"],
        ceiling_z_m=depth["best_height_m"],
        confidence={"depth": depth, "offset_ncc": offset["ncc"], "yaw_ncc": yaw["ncc"]},
    )


# ---------------------------------------------------------------------------
# Stage B: joint refinement by symmetric cross-position chi2


def _symmetric_cv_chi2(geom: GeometryConfig, cfg: RunConfig, ds: Dataset, cache_dir) -> float:
    tmaps = transmission_maps(ds, cfg.binning, cfg.calibration)
    omaps = {pid: opacity_map(t) for pid, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    fwd = build_forward_model(geom, first.txedges, first.tyedges, cache_dir=cache_dir)
    data = fit_data(fwd, omaps)
    rc = dataclasses.replace(cfg.reconstruction, algorithm="sirt", n_iter=60)
    scores = []
    for train in fwd.pose_ids:
        keep = np.zeros(fwd.A.shape[0], dtype=bool)
        keep[fwd.rows(train)] = True
        x, _ = solve(fwd, data.restricted(keep), rc)
        pred = fwd.predict_opacity(x)
        for test in fwd.pose_ids:
            if test == train:
                continue
            o = omaps[test]
            w = o.weights
            resid = o.lam - pred[test]
            sw = w.sum()
            if sw > 0:
                resid = resid - (w * resid).sum() / sw
            scores.append(float(np.sum(w * resid**2) / max(np.count_nonzero(w), 1)))
    return float(np.mean(scores)) if scores else np.inf


def stage_b(
    cfg: RunConfig,
    free_pose: str = "pos1",
    bounds_m: float = 0.5,
    bounds_deg: float = 10.0,
    cache_dir: str | None = None,
) -> GeometryEstimate:
    """Refine (dx, dy, yaw) of `free_pose` by minimizing symmetric cross-position chi2.

    Always searches around the config's own prior pose (`cfg.geometry`), never
    around Stage A's estimate -- see module docstring for why.
    """
    ds = load_dataset(cfg.data)
    base = cfg.geometry.pose(free_pose)
    x0 = np.array([base.x, base.y, base.yaw_deg])
    coarse = copy.deepcopy(cfg.geometry)
    coarse.grid_spacing_m = max(cfg.geometry.grid_spacing_m * 2, 0.2)

    def objective(theta):
        g = copy.deepcopy(coarse)
        g.poses[free_pose] = dataclasses.replace(base, x=theta[0], y=theta[1], yaw_deg=theta[2])
        return _symmetric_cv_chi2(g, cfg, ds, cache_dir)

    b = [(x0[0] - bounds_m, x0[0] + bounds_m), (x0[1] - bounds_m, x0[1] + bounds_m),
         (x0[2] - bounds_deg, x0[2] + bounds_deg)]
    res = optimize.minimize(objective, x0, method="Powell", bounds=b,
                            options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 40})
    return GeometryEstimate(
        dx=float(res.x[0]), dy=float(res.x[1]), yaw_deg=float(res.x[2]),
        ceiling_z_m=None,
        confidence={"final_chi2": float(res.fun), "n_eval": int(res.nfev), "converged": bool(res.success)},
    )


def refine_config(cfg: RunConfig, free_pose: str = "pos1", cache_dir: str | None = None) -> tuple[RunConfig, dict]:
    """Run Stage A (if real data, diagnostic only) then Stage B; return a refined RunConfig + report."""
    report: dict = {}
    if "phantom" not in cfg.data:
        report["stage_a"] = dataclasses.asdict(stage_a(cfg))
    final = stage_b(cfg, free_pose=free_pose, cache_dir=cache_dir)
    report["stage_b"] = dataclasses.asdict(final)

    out_cfg = copy.deepcopy(cfg)
    base = out_cfg.geometry.pose(free_pose)
    out_cfg.geometry.poses[free_pose] = dataclasses.replace(base, x=final.dx, y=final.dy, yaw_deg=final.yaw_deg)
    if final.ceiling_z_m:
        out_cfg.geometry.ceiling_z_prior_m = final.ceiling_z_m
    return out_cfg, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--free-pose", default="pos1")
    args = ap.parse_args(argv)
    cfg = RunConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    new_cfg, report = refine_config(cfg, free_pose=args.free_pose, cache_dir=str(out / "cache"))
    new_cfg.save(out / "config.json")
    (out / "selfcal_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report["stage_b"], indent=2))
    print(f"refined config -> {out}/config.json")


if __name__ == "__main__":
    main()

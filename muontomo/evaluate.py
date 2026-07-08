"""Evaluate a run directory: metrics.json scorecard + standard PNG set.

Pure with respect to the run inputs: reads volume.npz/config.json, writes only
metrics.json and images/. Usage:

    python -m muontomo.evaluate --run runs/exp01 [--truth phantoms/p1/truth_volume.npz]
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .calibration import transmission_maps
from .config import RunConfig, _git_hash
from .forward import build_forward_model
from .geometry import VoxelGrid
from .io import load_dataset
from .metrics import crossval as cv
from .metrics import fidelity, structure, truth as truth_m
from .metrics.volume import volume_stats
from .opacity import opacity_map
from .reconstruct import fit_data
from .render_png import render_run_images


def _load_volume(path: Path):
    with np.load(path) as z:
        return (
            z["rho"].astype(np.float64),
            VoxelGrid(origin=tuple(z["origin"]), spacing=float(z["spacing"]), shape=tuple(int(s) for s in z["shape"])),
            z["offsets"] if "offsets" in z else np.zeros(0),
            [str(p) for p in z["pose_ids"]] if "pose_ids" in z else [],
        )


def evaluate_run(run_dir: str | Path, truth_path: str | Path | None = None,
                 cache_dir: str | None = "runs/.cache") -> dict:
    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    ds = load_dataset(cfg.data)
    tmaps = transmission_maps(ds, cfg.binning, cfg.calibration)
    omaps = {pid: opacity_map(t) for pid, t in tmaps.items()}
    rho, grid, offsets, pose_ids = _load_volume(run / "volume.npz")
    fwd = build_forward_model(cfg.geometry, next(iter(tmaps.values())).txedges,
                              next(iter(tmaps.values())).tyedges, grid=grid, cache_dir=cache_dir)
    off = {pid: float(offsets[i]) for i, pid in enumerate(pose_ids)} if len(offsets) else {}
    lam_pred = fwd.predict_opacity(rho.ravel(), off)
    data = fit_data(fwd, omaps)

    card: dict = {
        "run": run.name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": _git_hash(),
        "config_hash": cfg.hash(),
        "algorithm": cfg.reconstruction.algorithm,
    }

    # ---- fidelity ----
    fid: dict = {}
    chi2s, devs, gaps = [], [], []
    for pid in fwd.pose_ids:
        o, t = omaps[pid], tmaps[pid]
        w = o.weights
        chi2 = fidelity.chi2_ndof(o.lam, lam_pred[pid], w)
        mu = t.scale * t.n_sky * np.exp(-lam_pred[pid])
        dev = fidelity.deviance_ndof(t.n_cafe, mu, t.mask)
        aligned, shift = fidelity.chi2_aligned(o.lam, lam_pred[pid], w)
        fid[pid] = {"chi2_ndof": chi2, "deviance_ndof": dev,
                    "chi2_aligned": aligned, "align_shift": list(shift)}
        chi2s.append(chi2)
        devs.append(dev)
        gaps.append(chi2 - aligned)
    fid["chi2_ndof"] = float(np.mean(chi2s))
    fid["deviance_ndof"] = float(np.mean(devs))
    fid["chi2_aligned_gap"] = float(np.mean(gaps))
    card["fidelity"] = fid

    # ---- cross-position validation ----
    cvd: dict = {}
    cv_chi2s, cv_rs = [], []
    for train in fwd.pose_ids:
        hv = run / f"volume_holdout_{train}.npz"
        if not hv.exists():
            continue
        rho_h, _, off_h, pids_h = _load_volume(hv)
        pred_h = fwd.predict_opacity(rho_h.ravel(), {})
        for test in fwd.pose_ids:
            if test == train:
                continue
            o = omaps[test]
            w = o.weights
            c = cv.heldout_chi2(o.lam, pred_h[test], w)
            r = cv.heldout_pearson(o.lam, pred_h[test], w)
            cvd[f"{train}->{test}"] = {"chi2": c, "pearson": r}
            cv_chi2s.append(c)
            cv_rs.append(r)
    if cv_chi2s:
        cvd["cv_chi2"] = float(np.mean(cv_chi2s))
        cvd["cv_pearson"] = float(np.mean(cv_rs))
        cvd["cv_gap"] = float(np.mean(cv_chi2s) - fid["chi2_ndof"])
    bh = run / "volume_binholdout.npz"
    keep_f = run / "binholdout_keep.npy"
    if bh.exists() and keep_f.exists():
        rho_b, _, off_b, pids_b = _load_volume(bh)
        off_bd = {pid: float(off_b[i]) for i, pid in enumerate(pids_b)} if len(off_b) else {}
        pred_b = fwd.predict_opacity(rho_b.ravel(), off_bd)
        keep = np.load(keep_f)
        chis = []
        for pid in fwd.pose_ids:
            o = omaps[pid]
            w = o.weights * (~keep[fwd.rows(pid)]).reshape(o.lam.shape)
            if (w > 0).sum() > 10:
                chis.append(cv.heldout_chi2(o.lam, pred_b[pid], w))
        if chis:
            cvd["binholdout_chi2"] = float(np.mean(chis))
    card["crossval"] = cvd

    # ---- structure ----
    zc = grid.axis_centers(2)
    pos = np.maximum(rho, 0.0)
    zprof = pos.sum(axis=(0, 1))
    iz = int(np.argmax(zprof)) if zprof.max() > 0 else rho.shape[2] // 2
    sl = pos[:, :, iz]
    st: dict = {"volume_slice": {**_structure_of(sl), "z_m": float(zc[iz])}}
    for pid in fwd.pose_ids:
        t = tmaps[pid]
        st[f"measured_{pid}"] = _structure_of(np.where(t.mask, t.T, np.nan), t.mask)
        st[f"pred_{pid}"] = _structure_of(np.where(t.mask, np.exp(-lam_pred[pid]), np.nan), t.mask)
    card["structure"] = st

    # ---- volume ----
    card["volume"] = volume_stats(rho, zc)

    # ---- truth (phantom only) ----
    tp = truth_path
    if tp is None and ds.is_phantom:
        cand = Path(ds.meta["dir"]) / "truth_volume.npz"
        tp = cand if cand.exists() else None
    if tp is not None:
        with np.load(tp) as z:
            tru = z["rho"].astype(np.float64)
            t_origin = z["origin"]
            t_sp = float(z["spacing"])
        tru_c = _crop_to_grid(tru, t_origin, t_sp, grid)
        if tru_c is not None:
            card["truth"] = {
                "rmse_scaled": truth_m.rmse_scaled(pos, tru_c),
                "ssim3d": truth_m.ssim3d(pos, tru_c),
                "iou_dense": truth_m.iou_dense(pos, tru_c),
                "z_error_m": truth_m.z_error_m(pos, tru_c, zc),
            }
        else:
            card["truth"] = None
    else:
        card["truth"] = None

    card["headline"] = {
        "chi2": round(fid["chi2_ndof"], 3),
        "cv_pearson": round(cvd.get("cv_pearson", float("nan")), 3),
        "z_peak_m": round(card["volume"]["z_peak_m"], 2),
        "z_width_m": round(card["volume"]["z_eff_width_m"], 2),
        "stripe_contrast": round(st["volume_slice"]["contrast"], 3),
    }
    if card["truth"]:
        card["headline"]["truth_ssim"] = round(card["truth"]["ssim3d"], 3)

    (run / "metrics.json").write_text(json.dumps(card, indent=2, default=float) + "\n")
    render_run_images(run, tmaps, lam_pred, omaps, rho, grid, card)
    return card


def _structure_of(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    per = structure.periodicity(img, mask)
    stripes = structure.stripe_stats(img, mask)
    return {
        "periodicity_snr": per["snr"],
        "period_bins": per["period_bins"],
        "angle_deg": per["angle_deg"],
        "contrast": stripes["contrast"],
        "cnr": stripes["cnr"],
        "flat_noise": structure.flat_noise(img, mask),
    }


def _crop_to_grid(tru: np.ndarray, t_origin, t_sp: float, grid: VoxelGrid):
    """Crop/align a truth volume (same spacing) onto the reconstruction grid."""
    if abs(t_sp - grid.spacing) > 1e-9:
        return None
    idx = []
    for ax in range(3):
        o = int(round((grid.origin[ax] - t_origin[ax]) / t_sp))
        if o < 0 or o + grid.shape[ax] > tru.shape[ax]:
            return None
        idx.append(slice(o, o + grid.shape[ax]))
    return tru[tuple(idx)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--truth", default=None)
    args = ap.parse_args(argv)
    card = evaluate_run(args.run, args.truth)
    print(json.dumps(card["headline"], indent=2))
    print(f"scorecard -> {args.run}/metrics.json ; images -> {args.run}/images/")


if __name__ == "__main__":
    main()

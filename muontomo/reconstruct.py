"""Reconstruction algorithms and the run-producing CLI.

All solvers share one linear model: lambda = A x + c_pose, where x >= 0 is the
voxel opacity density [1/m] and c_pose absorbs the per-pose normalization
nuisance. Fits are selected by row masks over the full system matrix, so the
same cached A serves the full fit, per-position cross-validation fits, and the
random-bin holdout fit.

Usage:
    python -m muontomo.reconstruct --config configs/production.json --out runs/exp01
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .calibration import transmission_maps
from .config import ReconstructionConfig, RunConfig
from .forward import ForwardModel, build_forward_model
from .io import Dataset, load_dataset
from .opacity import OpacityMap, opacity_map


@dataclass
class FitData:
    """Flattened measurements aligned with the rows of a ForwardModel."""

    lam: np.ndarray  # [n_rows]
    w: np.ndarray  # 1/sigma^2, 0 on masked-out bins
    pose_of_row: np.ndarray  # index into fwd.pose_ids per row

    def restricted(self, keep: np.ndarray) -> "FitData":
        w = np.where(keep, self.w, 0.0)
        return FitData(lam=self.lam, w=w, pose_of_row=self.pose_of_row)


def fit_data(fwd: ForwardModel, omaps: dict[str, OpacityMap]) -> FitData:
    lam = np.zeros(fwd.A.shape[0])
    w = np.zeros(fwd.A.shape[0])
    pose_of_row = np.zeros(fwd.A.shape[0], dtype=int)
    for i, pid in enumerate(fwd.pose_ids):
        rows = fwd.rows(pid)
        lam[rows] = omaps[pid].lam.ravel()
        w[rows] = omaps[pid].weights.ravel()
        pose_of_row[rows] = i
    return FitData(lam=lam, w=w, pose_of_row=pose_of_row)


def _update_offsets(resid: np.ndarray, w: np.ndarray, pose_of_row: np.ndarray, n_poses: int) -> np.ndarray:
    """Closed-form weighted-mean residual per pose (the c_pose nuisance)."""
    c = np.zeros(n_poses)
    for i in range(n_poses):
        sel = pose_of_row == i
        sw = w[sel].sum()
        if sw > 0:
            c[i] = (w[sel] * resid[sel]).sum() / sw
    return c


def _prepare(A: sparse.csr_matrix, w: np.ndarray):
    """Row/column scalings of the weighted system used by SIRT-type iterations."""
    Aw = A.multiply(np.sqrt(w)[:, None]).tocsr()
    row_sum = np.asarray(abs(Aw).sum(axis=1)).ravel()
    col_sum = np.asarray(abs(Aw).sum(axis=0)).ravel()
    return Aw, 1.0 / np.maximum(row_sum, 1e-12), 1.0 / np.maximum(col_sum, 1e-12)


def sirt(fwd: ForwardModel, data: FitData, rc: ReconstructionConfig) -> tuple[np.ndarray, dict]:
    """Weighted SIRT with nonnegativity and per-pose offset refinement."""
    A, lam, w = fwd.A, data.lam, data.w
    n_poses = len(fwd.pose_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_poses)
    _, row_inv, col_inv = _prepare(A, w)
    n_used = max(np.count_nonzero(w), 1)
    history = []
    for k in range(rc.n_iter):
        pred = A @ x + c[data.pose_of_row]
        resid = lam - pred
        c = c + _update_offsets(resid, w, data.pose_of_row, n_poses)
        resid = lam - (A @ x + c[data.pose_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if k % 20 == 0:
            history.append(chi2)
        if chi2 <= rc.chi2_target:  # discrepancy principle: stop at the noise floor
            break
        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
    history.append(chi2)
    return x, {"offsets": c.tolist(), "chi2_history": history, "n_iter_used": k + 1}


def _grad3(x3: np.ndarray, z_weight: float) -> np.ndarray:
    """Anisotropic forward differences -> [3, nx, ny, nz] (zero-padded at the far edge)."""
    g = np.zeros((3,) + x3.shape)
    g[0, :-1] = np.diff(x3, axis=0)
    g[1, :, :-1] = np.diff(x3, axis=1)
    g[2, :, :, :-1] = np.diff(x3, axis=2) * z_weight
    return g


def _div3(p: np.ndarray, z_weight: float) -> np.ndarray:
    """Negative adjoint of _grad3."""
    d = np.zeros(p.shape[1:])
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    d[:, :, :-1] += p[2, :, :, :-1] * z_weight
    d[:, :, 1:] -= p[2, :, :, :-1] * z_weight
    return d


def _prox_tv(v: np.ndarray, gamma: float, zw: float, y: np.ndarray, n_inner: int = 20) -> np.ndarray:
    """prox of gamma * ||grad .||_1 (anisotropic, nonneg-projected) at v.

    Chambolle dual projection: u = P_+(v + div y), y clipped to [-gamma, gamma].
    y is the warm-started dual state (mutated in place across outer iterations).
    """
    step = 1.0 / (4.0 * (2.0 + zw * zw))
    for _ in range(n_inner):
        u = np.maximum(v + _div3(y, zw), 0.0)
        np.clip(y + step * _grad3(u, zw), -gamma, gamma, out=y)
    return np.maximum(v + _div3(y, zw), 0.0)


def sirt_tv(fwd: ForwardModel, data: FitData, rc: ReconstructionConfig) -> tuple[np.ndarray, dict]:
    """SIRT iteration with a per-iteration TV proximal (denoising) step.

    The proximal-Landweber heuristic standard in practical tomography: keep
    SIRT's excellent preconditioning, apply edge-preserving anisotropic TV
    denoising each sweep. tv_alpha is the denoising threshold in units of the
    reconstructed opacity scale (its p95), so it transfers across datasets.
    Runs to n_iter and returns the iterate with the best chi2 (the TV step
    keeps it from overfitting the way plain SIRT does).
    """
    A, lam, w = fwd.A, data.lam, data.w
    shape = fwd.grid.shape
    n_poses = len(fwd.pose_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_poses)
    _, row_inv, col_inv = _prepare(A, w)
    n_used = max(np.count_nonzero(w), 1)
    dual = np.zeros((3,) + shape)
    history = []
    best = (np.inf, x.copy(), c.copy())
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.pose_of_row])
        c = c + _update_offsets(resid, w, data.pose_of_row, n_poses)
        resid = lam - (A @ x + c[data.pose_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if chi2 < best[0]:
            best = (chi2, x.copy(), c.copy())
        if k % 20 == 0:
            history.append(chi2)
        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
        gamma = rc.tv_alpha * max(float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 0.0, 1e-9)
        x = _prox_tv(x.reshape(shape), gamma, rc.tv_z_weight, dual, n_inner=10).ravel()
    history.append(best[0])
    return best[1], {"offsets": best[2].tolist(), "chi2_history": history, "best_chi2": best[0]}


def mlem_transmission(fwd: ForwardModel, data: FitData, rc: ReconstructionConfig,
                      counts: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, dict]:
    """ML-TR (Poisson transmission MLEM): n ~ Poisson(m * exp(-Ax)).

    counts[pid] = {"n": cafe counts, "m": expected open counts (scale * n_sky)}.
    Rows with zero weight (masked/holdout) are excluded from the update.
    """
    A = fwd.A
    n = np.zeros(A.shape[0])
    m = np.zeros(A.shape[0])
    for pid in fwd.pose_ids:
        rows = fwd.rows(pid)
        n[rows] = counts[pid]["n"].ravel()
        m[rows] = counts[pid]["m"].ravel()
    use = (data.w > 0) & (m > 0)
    n, m = np.where(use, n, 0.0), np.where(use, m, 0.0)
    x = np.zeros(A.shape[1])
    a1 = A @ np.ones(A.shape[1])
    s_pose = np.ones(len(fwd.pose_ids))  # per-pose normalization nuisance (ML update)
    history = []
    for k in range(rc.n_iter):
        att = np.exp(-np.clip(A @ x, 0, 50))
        mu = s_pose[data.pose_of_row] * m * att
        # closed-form ML update of the per-pose scales, then of the volume
        for i in range(len(s_pose)):
            sel = (data.pose_of_row == i) & use
            denom = np.sum(m[sel] * att[sel])
            if denom > 0:
                s_pose[i] = np.sum(n[sel]) / denom
        mu = s_pose[data.pose_of_row] * m * att
        num = A.T @ (mu - n)
        den = A.T @ (mu * a1)
        x = x + num / np.maximum(den, 1e-12)
        np.maximum(x, 0.0, out=x)
        if k % 20 == 0 or k == rc.n_iter - 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = 2 * np.sum(np.where(n > 0, mu - n + n * np.log(n / np.maximum(mu, 1e-300)), mu))
            dev_ndof = float(dev / max(use.sum(), 1))
            history.append(dev_ndof)
            if dev_ndof <= rc.chi2_target:  # discrepancy principle
                break
    return x, {"offsets": (-np.log(s_pose)).tolist(), "deviance_history": history}


def _layer_model(geom, txedges, tyedges, z_center: float, thickness: float, cache_dir=None) -> ForwardModel:
    """Forward model for a single thin horizontal layer at z_center."""
    from .config import GeometryConfig

    g = GeometryConfig(**{**vars(geom)})
    g.grid_z_m = (z_center - thickness / 2, z_center + thickness / 2)
    g.grid_spacing_m = geom.grid_spacing_m
    grid = None  # auto xy footprint at this height
    return build_forward_model(g, txedges, tyedges, grid=grid, cache_dir=cache_dir)


def layer_cv_score(
    geom,
    txedges: np.ndarray,
    tyedges: np.ndarray,
    omaps: dict,
    rc2: ReconstructionConfig,
    z: float,
    thickness: float,
    cache_dir=None,
    cv_trim_pct: float | None = None,
    active: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Cross-position validation score of a thin layer at height z.

    Fit the layer to each pose ALONE and score how well it predicts every other
    pose (free offset on the held-out view). Returns (cv_mean, per-pair dict).
    Shared by layered_fit's height selection and the autofocus scan
    (muontomo.focus) -- the scan calls this directly, skipping the full
    both-pose fit it does not need (1 of 3 solves per height).

    cv_trim_pct: if set (e.g. 90), each view's residual drops its worst
    (100 - cv_trim_pct)% of bins before averaging -- robust to the
    aliasing-induced outlier bins that spike the curve at isolated heights.
    """
    lf = _layer_model(geom, txedges, tyedges, float(z), thickness, cache_dir)
    ldata = fit_data(lf, omaps)
    if active is not None:
        ldata = ldata.restricted(active)
    cv: dict = {}
    scores = []
    for train in lf.pose_ids:
        keep = np.zeros(lf.A.shape[0], dtype=bool)
        keep[lf.rows(train)] = True
        x_tr, _ = sirt_tv(lf, ldata.restricted(keep), rc2)
        for test in lf.pose_ids:
            if test == train:
                continue
            rows = lf.rows(test)
            w = ldata.w[rows]
            resid = ldata.lam[rows] - (lf.A @ x_tr)[rows]
            sw = w.sum()
            if sw > 0:
                resid = resid - (w * resid).sum() / sw  # free offset on the held-out view
            r2 = w * resid**2
            if cv_trim_pct is not None:
                pos = r2[r2 > 0]
                thr = np.percentile(pos, cv_trim_pct) if pos.size else np.inf
                kept = r2 <= thr
                score = float(r2[kept].sum() / max(np.count_nonzero(kept), 1))
            else:
                score = float(np.sum(r2) / max(np.count_nonzero(w), 1))
            cv[f"{train}->{test}"] = score
            scores.append(score)
    return float(np.mean(scores)), cv


def layered_fit(
    fwd: ForwardModel,
    data: FitData,
    rc: ReconstructionConfig,
    geom,
    omaps,
    cache_dir=None,
    cv_trim_pct: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Height-scan layered reconstruction.

    For every candidate layer height, fit a thin single-layer 2D opacity map and
    score it by cross-position validation (fit one position, predict the other).
    The CV curve over height is the honest depth diagnostic in a 2-view setup
    (periodic ceiling patterns alias at discrete heights; CV + finite extent +
    angle-dependent focus break the tie). The best layer is embedded into the
    full 3D grid of `fwd` so downstream code sees an ordinary volume.

    cv_trim_pct: if set (e.g. 90), the CV residual per view drops the worst
    (100 - cv_trim_pct)% of bins before averaging. The plain mean-squared residual
    is dominated, at certain aliased heights, by a handful of pathological bins
    where the mis-registered beams land on gaps -- producing sharp spikes in the
    height curve. Trimming those outlier bins yields a smooth curve with the SAME
    minimum, and a height estimate that does not hinge on a few bins. Default None
    keeps the exact mean behaviour (unchanged for the reconstruction).
    """
    rc2 = ReconstructionConfig(**{**vars(rc), "algorithm": "tv", "tv_z_weight": 0.0})
    active = data.w > 0  # inherit any holdout/mask restriction (row layout is shared)
    scan = []
    for z_c in rc.layered_zs:
        lf = _layer_model(geom, fwd.txedges, fwd.tyedges, float(z_c), rc.layered_thickness, cache_dir)
        ldata = fit_data(lf, omaps).restricted(active)
        # full fit at this height (kept for the chi2 diagnostic)
        x_full, info_full = sirt_tv(lf, ldata, rc2)
        cv_mean, cv = layer_cv_score(
            geom, fwd.txedges, fwd.tyedges, omaps, rc2, float(z_c),
            rc.layered_thickness, cache_dir, cv_trim_pct, active,
        )
        scan.append({"z": float(z_c), "chi2": info_full["best_chi2"],
                     "cv": cv, "cv_mean": cv_mean})

    best = min(scan, key=lambda e: e["cv_mean"])
    lf = _layer_model(geom, fwd.txedges, fwd.tyedges, best["z"], rc.layered_thickness, cache_dir)
    ldata = fit_data(lf, omaps).restricted(active)
    x_layer, info = sirt_tv(lf, ldata, rc2)

    # embed the fitted layer into the full 3D grid
    x3 = np.zeros(fwd.grid.shape)
    layer2d = x_layer.reshape(lf.grid.shape)[:, :, 0]
    zc = fwd.grid.axis_centers(2)
    iz = int(np.argmin(np.abs(zc - best["z"])))
    xs = fwd.grid.axis_centers(0)
    ys = fwd.grid.axis_centers(1)
    lx = lf.grid.axis_centers(0)
    ly = lf.grid.axis_centers(1)
    i0 = np.searchsorted(xs - 1e-9, lx[0])
    j0 = np.searchsorted(ys - 1e-9, ly[0])
    ni = min(len(lx), len(xs) - i0)
    nj = min(len(ly), len(ys) - j0)
    # Spread the fitted layer across the number of z-voxels that actually span its
    # physical thickness, so the embedded structure has real volumetric extent
    # instead of being a single voxel with an inflated amplitude (which renders as
    # a flat sheet with no 3D shape). Each voxel keeps the fitted opacity density,
    # so total optical depth (opacity * spacing * n_vox) still matches the fitted
    # layer's opacity * layered_thickness.
    n_vox = max(1, int(round(rc.layered_thickness / fwd.grid.spacing)))
    iz0 = max(0, min(iz - n_vox // 2, fwd.grid.shape[2] - n_vox))
    iz1 = iz0 + n_vox
    x3[i0 : i0 + ni, j0 : j0 + nj, iz0:iz1] = layer2d[:ni, :nj, None]
    return x3.ravel(), {
        "offsets": info["offsets"],
        "chi2_history": info["chi2_history"],
        "best_chi2": info["best_chi2"],
        "layer_scan": scan,
        "layer_z": best["z"],
    }


SOLVERS = {"sirt": sirt, "tv": sirt_tv, "mlem": mlem_transmission}


def solve(fwd: ForwardModel, data: FitData, rc: ReconstructionConfig,
          counts: dict | None = None, *, geom=None, omaps: dict | None = None,
          cache_dir=None) -> tuple[np.ndarray, dict]:
    if rc.algorithm == "mlem":
        if counts is None:
            raise ValueError("mlem needs raw counts")
        return mlem_transmission(fwd, data, rc, counts)
    if rc.algorithm == "layered":
        if geom is None or omaps is None:
            raise ValueError("layered needs geom and omaps")
        return layered_fit(fwd, data, rc, geom, omaps, cache_dir)
    if rc.algorithm not in SOLVERS:
        raise ValueError(f"unknown algorithm {rc.algorithm!r}; have {sorted(SOLVERS)}")
    return SOLVERS[rc.algorithm](fwd, data, rc)


# ---------------------------------------------------------------------------
# Run production


def save_volume(path: Path, x: np.ndarray, fwd: ForwardModel, info: dict) -> None:
    np.savez_compressed(
        path,
        rho=x.reshape(fwd.grid.shape).astype(np.float32),
        origin=np.asarray(fwd.grid.origin),
        spacing=fwd.grid.spacing,
        shape=np.asarray(fwd.grid.shape),
        offsets=np.asarray(info.get("offsets", [])),
        pose_ids=np.asarray(fwd.pose_ids),
    )


def produce_run(cfg: RunConfig, out_dir: str | Path, cache_dir: str | None = "runs/.cache") -> Path:
    """Full fit + per-position CV fits + random-bin holdout fit -> a run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = load_dataset(cfg.data)
    tmaps = transmission_maps(ds, cfg.binning, cfg.calibration)
    omaps = {pid: opacity_map(t) for pid, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    fwd = build_forward_model(cfg.geometry, first.txedges, first.tyedges, cache_dir=cache_dir)
    data = fit_data(fwd, omaps)
    counts = {
        pid: {"n": t.n_cafe, "m": t.scale * t.n_sky} for pid, t in tmaps.items()
    }

    rc = cfg.reconstruction
    rng = np.random.default_rng(rc.seed)
    info: dict = {"algorithm": rc.algorithm, "fits": {}}

    def run_fit(tag: str, keep: np.ndarray) -> None:
        x, fit_info = solve(fwd, data.restricted(keep), rc, counts,
                            geom=cfg.geometry, omaps=omaps, cache_dir=cache_dir)
        fit_info["n_rows_used"] = int(np.count_nonzero(data.w * keep))
        info["fits"][tag] = fit_info
        save_volume(out / f"volume{'' if tag == 'full' else '_' + tag}.npz", x, fwd, fit_info)

    all_rows = np.ones(fwd.A.shape[0], dtype=bool)
    run_fit("full", all_rows)
    for pid in fwd.pose_ids:
        only = np.zeros_like(all_rows)
        only[fwd.rows(pid)] = True
        run_fit(f"holdout_{pid}", only)  # trained on this position ONLY
    binmask = rng.random(fwd.A.shape[0]) >= rc.holdout_fraction
    run_fit("binholdout", binmask)
    np.save(out / "binholdout_keep.npy", binmask)

    info["runtime_s"] = round(time.time() - t0, 2)
    info["scale"] = {pid: t.scale for pid, t in tmaps.items()}
    info["grid"] = {"origin": fwd.grid.origin, "spacing": fwd.grid.spacing, "shape": fwd.grid.shape}
    (out / "fit_info.json").write_text(json.dumps(info, indent=2) + "\n")
    cfg.save(out / "config.json")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = produce_run(RunConfig.load(args.config), args.out)
    print(f"run written to {out}")


if __name__ == "__main__":
    main()

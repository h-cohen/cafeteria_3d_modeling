"""Model-free autofocus: recover the beams' height from two-view parallax.

Plane-sweep / shape-from-focus. Backproject each detector's measured opacity
(-ln T) onto a horizontal plane at a candidate height z. At the TRUE height the
two detectors' beam patterns register on top of one another, so the two images
AGREE (high normalized cross-correlation) and the summed refocus is SHARP; at a
wrong height parallax shifts them apart and both metrics fall. Sweeping z and
taking the peak is exactly how a light-field camera or tomosynthesis stack
refocuses -- and it needs two views, which is why a single detector cannot do it.

Two families of metric, of different trustworthiness:
  * AUTHORITATIVE -- a model-based cross-validation height-scan (cv_height_scan):
    fit a thin ceiling layer to ONE detector, score how well it predicts the
    OTHER. Only at the true height can one layer explain both views, and the
    finite room breaks the periodic-beam aliasing that fools naive correlation.
    Validated against synthetic ground truth (see scripts/autofocus_validation.py).
  * QUICK-LOOK -- a model-free two-view NCC curve + per-region height map. Fast,
    but biased by large non-beam features and prone to parallax aliasing; used
    for a fast sanity check and the viewer HUD, not as the final answer.

run_focus() writes focus.json, images/focus.png, and a 2-page autofocus_report.pdf
(a plain-language explanation + this run's result), regenerated every evaluate.

    python -m muontomo.focus --run runs/production
"""

from __future__ import annotations

import numpy as np

from .backproject import backproject_opacity


def _interior_mask(xs: np.ndarray, ys: np.ndarray, margin_m: float = 1.0) -> np.ndarray:
    """Boolean [nx, ny] dropping a margin off every edge (limited-angle backprojection
    pools spurious mass at the boundary; keep the scan on the trustworthy centre)."""
    mx = (xs >= xs[0] + margin_m) & (xs <= xs[-1] - margin_m)
    my = (ys >= ys[0] + margin_m) & (ys <= ys[-1] - margin_m)
    return np.outer(mx, my)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two images over `mask` (scale-free)."""
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    x, y = a[m], b[m]
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def _sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """Normalized gradient energy of the refocused stack (scale-free): high when crisp."""
    m = mask & np.isfinite(img)
    if m.sum() < 20:
        return np.nan
    im = np.where(m, img, np.nan)
    gx, gy = np.gradient(np.nan_to_num(im, nan=float(np.nanmean(im))))
    var = float(np.nanvar(im))
    return float(np.nanmean(gx**2 + gy**2) / var) if var > 0 else np.nan


def _local_ncc(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    """Per-pixel windowed NCC via box filters; NaN where local coverage is thin."""
    from scipy.ndimage import uniform_filter

    ok = np.isfinite(a) & np.isfinite(b)
    A = np.where(ok, a, 0.0)
    B = np.where(ok, b, 0.0)
    n = uniform_filter(ok.astype(float), size, mode="constant")
    nn = np.maximum(n, 1e-6)
    ma = uniform_filter(A, size, mode="constant") / nn
    mb = uniform_filter(B, size, mode="constant") / nn
    saa = uniform_filter(A * A, size, mode="constant") / nn - ma**2
    sbb = uniform_filter(B * B, size, mode="constant") / nn - mb**2
    sab = uniform_filter(A * B, size, mode="constant") / nn - ma * mb
    denom = np.sqrt(np.maximum(saa, 0) * np.maximum(sbb, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = np.where(denom > 0, sab / np.maximum(denom, 1e-12), np.nan)
    ncc[n < 0.3] = np.nan  # <30% of the window covered -> no reliable correlation
    return ncc


def _pair_stacks(tmaps: dict, geom, zs: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Backproject the two detectors onto every candidate plane once.

    Returns (a_stack, b_stack, mean_stack) each [len(zs), nx, ny]. Uses the first
    two poses (the two detectors); with >2 poses the rest are ignored for the
    pairwise parallax metric.
    """
    pids = list(tmaps)[:2]
    a_list, b_list, m_list = [], [], []
    for z in zs:
        per_pose, mean = backproject_opacity(tmaps, geom, float(z), xs, ys)
        a_list.append(per_pose[pids[0]])
        b_list.append(per_pose[pids[1]])
        m_list.append(mean)
    return np.stack(a_list), np.stack(b_list), np.stack(m_list)


def focus_curve(a_stack, b_stack, mean_stack, zs, xs, ys):
    """Agreement (NCC) and sharpness vs height. Returns a dict; best_z from the NCC peak."""
    interior = _interior_mask(xs, ys)
    ncc = np.array([_ncc(a_stack[k], b_stack[k], interior) for k in range(len(zs))])
    sharp = np.array([_sharpness(mean_stack[k], interior) for k in range(len(zs))])
    i = int(np.nanargmax(ncc))
    js = int(np.nanargmax(sharp)) if np.isfinite(sharp).any() else i
    return {
        "zs_m": [float(z) for z in zs],
        "ncc": [None if not np.isfinite(v) else round(float(v), 4) for v in ncc],
        "sharpness": [None if not np.isfinite(v) else round(float(v), 4) for v in sharp],
        "focus_z_m": round(float(zs[i]), 3),
        "focus_ncc": round(float(ncc[i]), 4),
        "sharpness_z_m": round(float(zs[js]), 3),
    }


def height_map(a_stack, b_stack, zs, xs, ys, window_m: float = 1.0, min_conf: float = 0.2):
    """Windowed autofocus: best-agreement height per pixel (shape-from-focus).

    Returns (z_map [nx, ny] with NaN where unreliable, conf_map, stats dict)."""
    res = float(xs[1] - xs[0])
    size = max(3, int(round(window_m / res)))
    nx, ny = a_stack.shape[1:]
    best_z = np.full((nx, ny), np.nan)
    best_c = np.full((nx, ny), -np.inf)
    for k, z in enumerate(zs):
        c = _local_ncc(a_stack[k], b_stack[k], size)
        upd = np.isfinite(c) & (c > best_c)
        best_z[upd] = float(z)
        best_c[upd] = c[upd]
    best_z[~np.isfinite(best_c) | (best_c < min_conf)] = np.nan
    vals = best_z[np.isfinite(best_z)]
    stats = {
        "window_m": window_m,
        "median_z_m": round(float(np.median(vals)), 3) if vals.size else None,
        "iqr_z_m": round(float(np.subtract(*np.percentile(vals, [75, 25]))), 3) if vals.size else None,
        "coverage_frac": round(float(vals.size / best_z.size), 3),
    }
    return best_z, np.where(np.isfinite(best_c), best_c, np.nan), stats


def _load_run(run_dir):
    from pathlib import Path

    from .calibration import transmission_maps
    from .config import RunConfig
    from .io import load_dataset

    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    return run, cfg, tmaps


def _grid(cfg, res_m: float):
    (x0, x1), (y0, y1) = cfg.geometry.grid_xy_m
    xs = np.arange(x0 + res_m / 2, x1, res_m)
    ys = np.arange(y0 + res_m / 2, y1, res_m)
    return xs, ys


def _default_zs(cfg):
    """Sweep a generous band around the configured layer (or 4-9 m if unset)."""
    if cfg.reconstruction.layered_zs:
        z0 = float(cfg.reconstruction.layered_zs[0])
        lo, hi = z0 - 2.0, z0 + 2.0
    else:
        lo, hi = 4.0, 9.0
    zmin = max(lo, max(cfg.geometry.pose(p).z for p in cfg.geometry.poses) + 0.5)
    return np.round(np.arange(zmin, hi + 1e-6, 0.1), 3)


def quick_focus(tmaps, geom, cfg, res_m: float = 0.15) -> dict:
    """Coarse model-free NCC scan (the fast 'quick-look'); no height map, no CV.

    Used by the viewer build so opening the viewer stays fast; the authoritative
    cross-validated height comes from run_focus (written to focus.json)."""
    zs = _default_zs(cfg)
    xs, ys = _grid(cfg, res_m)
    a, b, m = _pair_stacks(tmaps, geom, zs, xs, ys)
    return focus_curve(a, b, m, zs, xs, ys)


def _scan_zs(cfg) -> np.ndarray:
    """Candidate heights for the CV scan: the assumed layer height +/- 1 m at 0.1 m
    (a fine grid so the CV-vs-height curve is smooth around its minimum)."""
    if cfg.reconstruction.layered_zs:
        z0 = float(cfg.reconstruction.layered_zs[0])
    else:
        z0 = float(getattr(cfg.geometry, "ceiling_z_prior_m", 7.0))
    zmin = max(z0 - 1.0, max(cfg.geometry.pose(p).z for p in cfg.geometry.poses) + 1.0)
    return np.round(np.arange(zmin, z0 + 1.0 + 1e-6, 0.1), 2)


def cv_height_scan(cfg, tmaps, zs, cache_dir=None):
    """Authoritative, alias-robust autofocus: for each candidate height fit a thin
    ceiling layer to ONE detector and score how well it predicts the OTHER
    (cross-position validation, via reconstruct.layered_fit). The finite room and
    the physics model break the periodic-beam parallax aliasing that fools the
    naive correlation. Returns (scan_list, best_z) where scan_list entries are
    {z, cv_mean, chi2}; best_z minimizes cv_mean.
    """
    from dataclasses import replace

    from .forward import build_forward_model
    from .opacity import opacity_map
    from .reconstruct import fit_data, layered_fit

    omaps = {p: opacity_map(t) for p, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    fwd = build_forward_model(cfg.geometry, first.txedges, first.tyedges, cache_dir=cache_dir)
    data = fit_data(fwd, omaps)
    rc = replace(cfg.reconstruction, algorithm="layered",
                 layered_zs=tuple(float(z) for z in zs), tv_z_weight=0.0)
    # Trim the worst 10% of residual bins per view: robust to the aliasing-induced
    # outlier bins that otherwise spike the CV curve at isolated wrong heights.
    _, info = layered_fit(fwd, data, rc, cfg.geometry, omaps, cache_dir=cache_dir,
                          cv_trim_pct=90.0)
    scan = [{"z": float(e["z"]), "cv_mean": float(e["cv_mean"]), "chi2": float(e["chi2"])}
            for e in info["layer_scan"]]
    best = min(scan, key=lambda e: e["cv_mean"])
    return scan, float(best["z"])


def run_focus(run_dir, res_m: float = 0.1, window_m: float = 1.0, report: bool = True) -> dict:
    """Full autofocus for a run. Produces:

      * the AUTHORITATIVE height from the cross-validation height-scan (alias-robust),
      * the model-free NCC 'quick-look' curve + per-region height map,
      * focus.json, images/focus.png, and a 2-page autofocus_report.pdf.
    """
    import json
    from pathlib import Path

    run, cfg, tmaps = _load_run(run_dir)
    if len(tmaps) < 2:
        raise ValueError("autofocus needs >= 2 detectors (parallax)")

    # model-free quick-look: NCC curve + height map
    zs = _default_zs(cfg)
    xs, ys = _grid(cfg, res_m)
    a, b, m = _pair_stacks(tmaps, cfg.geometry, zs, xs, ys)
    curve = focus_curve(a, b, m, zs, xs, ys)
    zmap, conf, stats = height_map(a, b, zs, xs, ys, window_m=window_m)

    # authoritative model-based cross-validation height-scan
    cache_dir = str(Path(run).parent / ".cache")
    scan, cv_z = cv_height_scan(cfg, tmaps, _scan_zs(cfg), cache_dir=cache_dir)

    z0 = float(cfg.reconstruction.layered_zs[0]) if cfg.reconstruction.layered_zs else cv_z
    result = {
        "autofocus_z_m": round(cv_z, 2),
        "autofocus_method": "cross-validation height-scan (fit one detector, predict the other)",
        "cv_scan": [{"z": e["z"], "cv_mean": round(e["cv_mean"], 4),
                     "chi2": round(e["chi2"], 4)} for e in scan],
        "quicklook_z_m": curve["focus_z_m"],
        "quicklook_ncc": curve["focus_ncc"],
        "ncc_curve": {"zs_m": curve["zs_m"], "ncc": curve["ncc"]},
        "height_map": stats,
        "solve_z_m": round(z0, 2),
        "report_pdf": "autofocus_report.pdf",
    }
    (run / "focus.json").write_text(json.dumps(result, indent=2))
    _render_focus_png(run, curve, xs, ys, zmap, conf, stats)
    if report:
        try:
            from .focus_report import render_report

            # red/cyan parallax overlay at three heights around the CV pick
            ref_z = [round(cv_z - 1.0, 2), round(cv_z, 2), round(cv_z + 1.0, 2)]
            ai = [int(np.argmin(np.abs(zs - z))) for z in ref_z]
            render_report(run, cfg, result, xs, ys, ref_z,
                          np.stack([a[i] for i in ai]), np.stack([b[i] for i in ai]))
        except Exception as e:
            print(f"note: autofocus report not generated ({e})")
    return result


def _render_focus_png(run, curve, xs, ys, zmap, conf, stats) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = np.array(curve["zs_m"])
    ncc = np.array([np.nan if v is None else v for v in curve["ncc"]])
    sharp = np.array([np.nan if v is None else v for v in curve["sharpness"]])
    zf = curve["focus_z_m"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax0.plot(zs, ncc, "-o", ms=3, color="#1f77b4", label="two-view agreement (NCC)")
    ax0.axvline(zf, color="crimson", ls="--", label=f"focus z* = {zf:.2f} m")
    ax0.set(xlabel="candidate plane height z (m)", ylabel="two-view NCC", title="Autofocus curve")
    axt = ax0.twinx()
    axt.plot(zs, sharp, "-", color="#999", alpha=0.7, label="refocus sharpness")
    axt.set_ylabel("sharpness (norm. gradient energy)", color="#777")
    ax0.legend(loc="lower center", fontsize=9)

    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    finite = zmap[np.isfinite(zmap)]
    if finite.size:
        vlo, vhi = np.percentile(finite, [5, 95])
    else:
        vlo, vhi = zf - 0.5, zf + 0.5
    im = ax1.imshow(zmap.T, origin="lower", extent=ext, cmap="turbo", vmin=vlo, vmax=vhi)
    med = stats.get("median_z_m")
    ax1.set(xlabel="x (m)", ylabel="y (m)",
            title=f"Height map (median {med} m, IQR {stats.get('iqr_z_m')} m)")
    plt.colorbar(im, ax=ax1, fraction=0.046, label="autofocus height z (m)")

    img_dir = Path(run) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "focus.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--window", type=float, default=1.0, help="height-map window (m)")
    args = ap.parse_args(argv)
    r = run_focus(args.run, res_m=args.res, window_m=args.window)
    print(f"autofocus height (cross-validated) = {r['autofocus_z_m']} m  |  "
          f"quick-look NCC = {r['quicklook_z_m']} m  |  "
          f"report -> {args.run}/{r['report_pdf']}")


if __name__ == "__main__":
    main()

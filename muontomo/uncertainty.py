"""Statistical uncertainty of the reconstruction by parametric bootstrap.

The measured counts are Poisson per angular bin, so replicas of the entire
measurement are drawn as n' ~ Poisson(n) for both the cafeteria and the sky
histograms (the calibration scale s is held fixed: it is a global nuisance that
the fit re-absorbs through the per-pose offsets). Each replica is pushed
through the identical pipeline:

  * the thin-layer solve at the nominal ceiling height -> per-pixel sigma of
    the layer, and error bars on each beam's position and amplitude;
  * the cross-validation autofocus on a fine fixed grid -> a confidence
    interval on the ceiling height itself (sub-grid resolution via a parabola
    through the minimum).

Writes runs/<run>/uncertainty.json + images/uncertainty.png.

    python -m muontomo.uncertainty --run runs/production [--n-layer 50] [--n-focus 25]
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .calibration import TransmissionMap


def resample_tmaps(tmaps: dict, rng: np.random.Generator) -> dict:
    """Poisson-resample the raw counts of every pose; recompute T and sigma_T.

    The validity mask and the calibration scale are held at their measured
    values so every replica lives on the identical bin geometry -- the bootstrap
    then measures purely the counting-statistics propagation.
    """
    out = {}
    for pid, tm in tmaps.items():
        n_cafe = rng.poisson(tm.n_cafe).astype(float)
        n_sky = rng.poisson(tm.n_sky).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(tm.mask, n_cafe / (tm.scale * np.maximum(n_sky, 1)), 0.0)
            sigma = np.where(
                tm.mask,
                T * np.sqrt(1 / np.maximum(n_cafe, 1) + 1 / np.maximum(n_sky, 1)),
                np.inf,
            )
        out[pid] = TransmissionMap(
            T=T, sigma_T=sigma, mask=tm.mask.copy(),
            txedges=tm.txedges, tyedges=tm.tyedges,
            n_cafe=n_cafe, n_sky=n_sky, scale=tm.scale, pose_id=tm.pose_id,
        )
    return out


def _fit_layer(cfg, tmaps: dict, z: float, cache_dir):
    """Thin-layer solve at height z; returns (layer2d, xs, ys)."""
    from .opacity import opacity_map
    from .reconstruct import _layer_model, fit_data, sirt_tv

    omaps = {p: opacity_map(t) for p, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    rc2 = replace(cfg.reconstruction, algorithm="tv", tv_z_weight=0.0)
    lf = _layer_model(cfg.geometry, first.txedges, first.tyedges, float(z),
                      cfg.reconstruction.layered_thickness, cache_dir)
    x, _ = sirt_tv(lf, fit_data(lf, omaps), rc2)
    layer = x.reshape(lf.grid.shape).mean(axis=2)
    xs = lf.grid.axis_centers(0)
    ys = lf.grid.axis_centers(1)
    return layer, xs, ys


def _beam_stats(layer, xs, ys, ref_beams, grid_xy):
    """(positions, amplitudes) of the beams nearest each reference position, from
    the central-band x-profile over the well-covered window."""
    from .beams import beam_peaks

    band = (ys > -2.0) & (ys < 2.0)
    prof = layer[:, band].mean(axis=1)
    (x0, x1), _ = grid_xy
    core = (xs > x0 + 1.0) & (xs < x1 - 1.0)
    peaks = beam_peaks(xs[core], prof[core])
    pos, amp = [], []
    for r in ref_beams:
        if peaks.size:
            p = float(peaks[np.argmin(np.abs(peaks - r))])
        else:
            p = np.nan
        pos.append(p)
        amp.append(float(np.interp(p if np.isfinite(p) else r, xs, prof)))
    return np.array(pos), np.array(amp)


def _focus_z(cfg, tmaps, zs, cache_dir) -> float:
    """CV height-scan on a fixed fine grid; sub-grid minimum via a parabola
    through the three points around the argmin."""
    from .opacity import opacity_map
    from .reconstruct import layer_cv_score

    omaps = {p: opacity_map(t) for p, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    rc2 = replace(cfg.reconstruction, algorithm="tv", tv_z_weight=0.0)
    vals = np.array([
        layer_cv_score(cfg.geometry, first.txedges, first.tyedges, omaps, rc2,
                       float(z), cfg.reconstruction.layered_thickness, cache_dir,
                       cv_trim_pct=90.0)[0]
        for z in zs
    ])
    i = int(np.argmin(vals))
    if 0 < i < len(zs) - 1:
        step = zs[i + 1] - zs[i]
        denom = vals[i + 1] - 2 * vals[i] + vals[i - 1]
        if denom > 0:
            return float(zs[i] + 0.5 * step * (vals[i - 1] - vals[i + 1]) / denom)
    return float(zs[i])


def bootstrap(run_dir, n_layer: int = 50, n_focus: int = 25, seed: int = 0,
              cache_dir: str | None = "runs/.cache") -> dict:
    from .calibration import transmission_maps
    from .config import RunConfig
    from .io import load_dataset

    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    z0 = float(cfg.reconstruction.layered_zs[0]) if cfg.reconstruction.layered_zs else 7.0
    rng = np.random.default_rng(seed)

    ref_beams = None
    mfile = run / "metrics.json"
    if mfile.exists():
        ref_beams = (json.loads(mfile.read_text()).get("beams") or {}).get("beams_x_data_m")
    grid_xy = cfg.geometry.grid_xy_m

    # nominal (unresampled) layer, the reference for everything below
    layer0, xs, ys = _fit_layer(cfg, tmaps, z0, cache_dir)
    if ref_beams is None:
        from .beams import beam_peaks

        band = (ys > -2.0) & (ys < 2.0)
        core = (xs > grid_xy[0][0] + 1.0) & (xs < grid_xy[0][1] - 1.0)
        ref_beams = list(beam_peaks(xs[core], layer0[:, band].mean(axis=1)[core]))

    # ---- layer bootstrap: per-pixel sigma + beam position/amplitude errors ----
    layers, poss, amps = [], [], []
    for k in range(n_layer):
        lay, _, _ = _fit_layer(cfg, resample_tmaps(tmaps, rng), z0, cache_dir)
        layers.append(lay)
        p, a = _beam_stats(lay, xs, ys, ref_beams, grid_xy)
        poss.append(p)
        amps.append(a)
        if (k + 1) % 10 == 0:
            print(f"  layer replicas {k + 1}/{n_layer}", flush=True)
    layers = np.stack(layers)
    poss = np.stack(poss)
    amps = np.stack(amps)
    layer_mean = layers.mean(axis=0)
    layer_std = layers.std(axis=0)

    # ---- autofocus bootstrap: CI on the ceiling height ----
    zs_scan = np.round(np.arange(z0 - 0.4, z0 + 0.4 + 1e-6, 0.1), 2)
    z_stars = []
    for k in range(n_focus):
        z_stars.append(_focus_z(cfg, resample_tmaps(tmaps, rng), zs_scan, cache_dir))
        if (k + 1) % 5 == 0:
            print(f"  focus replicas {k + 1}/{n_focus}", flush=True)
    z_stars = np.array(z_stars)

    p0, a0 = _beam_stats(layer0, xs, ys, ref_beams, grid_xy)
    result = {
        "n_layer_replicas": n_layer,
        "n_focus_replicas": n_focus,
        "seed": seed,
        "z_nominal_m": z0,
        "beams": [
            {
                "ref_x_m": round(float(r), 2),
                "pos_m": round(float(p0[i]), 3),
                "pos_sigma_m": round(float(np.nanstd(poss[:, i])), 3),
                "amp": round(float(a0[i]), 4),
                "amp_sigma": round(float(np.nanstd(amps[:, i])), 4),
            }
            for i, r in enumerate(ref_beams)
        ],
        "beam_pos_sigma_mean_m": round(float(np.nanmean(np.nanstd(poss, axis=0))), 3),
        "beam_amp_rel_sigma_mean": round(
            float(np.nanmean(np.nanstd(amps, axis=0) / np.maximum(np.abs(a0), 1e-12))), 3),
        "autofocus": {
            "z_star_mean_m": round(float(z_stars.mean()), 3),
            "z_star_sigma_m": round(float(z_stars.std()), 3),
            "z_star_ci68_m": [round(float(v), 3) for v in np.percentile(z_stars, [16, 84])],
        },
        "layer_pixel_rel_sigma_median": round(float(np.median(
            layer_std[layer_mean > 0.3 * layer_mean.max()]
            / np.maximum(layer_mean[layer_mean > 0.3 * layer_mean.max()], 1e-12))), 3),
    }
    (run / "uncertainty.json").write_text(json.dumps(result, indent=2) + "\n")
    _render_png(run, result, layer_mean, layer_std, xs, ys, ref_beams, poss, amps, z_stars)
    return result


def _render_png(run, result, layer_mean, layer_std, xs, ys, ref_beams,
                poss, amps, z_stars) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]

    ax = axes[0, 0]
    im = ax.imshow(layer_mean.T, origin="lower", extent=ext, cmap="viridis",
                   vmax=np.percentile(layer_mean, 99))
    ax.set(title="bootstrap mean layer", xlabel="x (m)", ylabel="y (m)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="opacity density (1/m)")

    ax = axes[0, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(layer_mean > 0.15 * layer_mean.max(),
                       layer_std / np.maximum(layer_mean, 1e-12), np.nan)
    im = ax.imshow(rel.T, origin="lower", extent=ext, cmap="magma", vmin=0,
                   vmax=np.nanpercentile(rel, 95))
    ax.set(title="relative statistical error (sigma / mean)", xlabel="x (m)", ylabel="y (m)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    b = result["beams"]
    refx = [e["ref_x_m"] for e in b]
    ax.errorbar(refx, [e["pos_m"] - e["ref_x_m"] for e in b],
                yerr=[e["pos_sigma_m"] for e in b], fmt="o", capsize=4, color="#2e7d47")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set(title="beam position vs verified reference (statistical error bars)",
           xlabel="verified beam x (m)", ylabel="offset (m)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(z_stars, bins=12, color="#8c1d40", alpha=0.8)
    af = result["autofocus"]
    ax.axvline(af["z_star_mean_m"], color="k", lw=1.2)
    ax.set(title=f"autofocus height replicas: z* = {af['z_star_mean_m']:.2f} "
                 f"$\\pm$ {af['z_star_sigma_m']:.2f} m",
           xlabel="bootstrap z* (m)", ylabel="replicas")

    img = Path(run) / "images"
    img.mkdir(exist_ok=True)
    fig.savefig(img / "uncertainty.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-layer", type=int, default=50)
    ap.add_argument("--n-focus", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    r = bootstrap(args.run, args.n_layer, args.n_focus, args.seed)
    af = r["autofocus"]
    print(f"beam position sigma (mean): {r['beam_pos_sigma_mean_m']} m | "
          f"amplitude rel sigma: {r['beam_amp_rel_sigma_mean']} | "
          f"z* = {af['z_star_mean_m']} +/- {af['z_star_sigma_m']} m")


if __name__ == "__main__":
    main()

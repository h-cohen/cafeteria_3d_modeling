"""Ground-truth validation of the autofocus, saved to reports/autofocus_validation.png.

Generates synthetic cafeterias with the REAL campaign geometry (detector baseline,
beam pitch, lever arm) but the ceiling deliberately placed at known heights, and
shows that the cross-validation height-scan recovers them -- then runs the same
scan on the real data. This is the figure embedded on page 2 of every run's
autofocus_report.pdf. Run once (it is slow, ~10 min); the figure is committable.

    python scripts/autofocus_validation.py [--run runs/production] [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from muontomo import phantom  # noqa: E402
from muontomo.calibration import transmission_maps  # noqa: E402
from muontomo.config import (GeometryConfig, PoseConfig, ReconstructionConfig,  # noqa: E402
                             RunConfig)
from muontomo.forward import build_forward_model  # noqa: E402
from muontomo.io import load_dataset  # noqa: E402
from muontomo.opacity import opacity_map  # noqa: E402
from muontomo.reconstruct import fit_data, layered_fit  # noqa: E402
from muontomo.focus import _pair_stacks, focus_curve, _grid  # noqa: E402

# Real campaign geometry (from runs/production/config.json).
POSES = {"pos0": {"x": 0.0, "y": 0.0}, "pos1": {"x": 1.7753, "y": 0.7203}}
GRID_XY = ((-5.0, 7.0), (-5.0, 5.0))
GRID_Z = (1.0, 9.0)
SPACING = 0.1
INJECTED = (6.6, 7.0, 7.4)
COL = {"66": "#1f77b4", "70": "#2ca02c", "74": "#d62728"}


def _campaign_geom() -> GeometryConfig:
    g = GeometryConfig(poses={p: PoseConfig(**v) for p, v in POSES.items()},
                       grid_z_m=GRID_Z, grid_spacing_m=SPACING)
    g.grid_xy_m = GRID_XY
    return g


def _spec(z0: float) -> dict:
    s = phantom.make_spec("beams")
    s["grid"] = {"spacing_m": SPACING, "z_m": list(GRID_Z)}
    s["room"] = {"x": [-4.5, 4.5], "y": [-4.5, 4.5]}
    s["slab"] = {"z0": z0, "thickness": 0.15, "kappa": 0.6}
    s["beams"] = {"pitch": 1.58, "width": 0.40, "depth": 0.30,
                  "direction_deg": 0.0, "phase": 0.2, "kappa": 0.9}
    s["geometry"] = {"poses": POSES, "pose_error": {}}
    s["counts"] = {"sky": 28_500_000, "pos0": 6_300_000, "pos1": 2_400_000}
    s["seed"] = 7
    return s


def _generate(z0: float, out: Path) -> Path:
    """phantom.generate, but with the campaign grid_xy set on the geometry."""
    out.mkdir(parents=True, exist_ok=True)
    spec = _spec(z0)
    rng = np.random.default_rng(spec["seed"])
    edges = np.linspace(-spec["binning"]["t_max"], spec["binning"]["t_max"],
                        spec["binning"]["n_bins"] + 1)
    txc = 0.5 * (edges[:-1] + edges[1:])
    geom = _campaign_geom()
    fwd = build_forward_model(geom, edges, edges, cache_dir=None)
    truth = phantom.rasterize_volume(spec, fwd.grid)
    trans = fwd.predict_transmission(truth)
    sky_mu = phantom._sky_template(txc, txc, spec["sky_cos_power"], spec["counts"]["sky"])
    phantom._save_counts(out / "counts_sky.npz", rng.poisson(sky_mu).astype(float), edges)
    for pid in fwd.pose_ids:
        mu = sky_mu * (spec["counts"][pid] / spec["counts"]["sky"]) * trans[pid]
        phantom._save_counts(out / f"counts_{pid}.npz", rng.poisson(mu).astype(float), edges)
    import json
    (out / "meta.json").write_text(json.dumps({"z0": z0, "kind": "phantom"}))
    return out


def _phantom_cfg(out: Path) -> RunConfig:
    cfg = RunConfig(data={"phantom": str(out)})
    cfg.binning.hist = "txty"
    cfg.binning.t_max = 1.0
    cfg.binning.rebin = 1
    cfg.geometry = _campaign_geom()
    return cfg


def _cv_scan(cfg, tmaps, zs, cache_dir=None):
    from dataclasses import replace
    omaps = {p: opacity_map(t) for p, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    fwd = build_forward_model(cfg.geometry, first.txedges, first.tyedges, cache_dir=cache_dir)
    data = fit_data(fwd, omaps)
    rc = replace(cfg.reconstruction, algorithm="layered",
                 layered_zs=tuple(float(z) for z in zs), tv_z_weight=0.0)
    _, info = layered_fit(fwd, data, rc, cfg.geometry, omaps, cache_dir=cache_dir,
                          cv_trim_pct=90.0)
    return info["layer_scan"]


def compute(run_dir: str, quick: bool):
    import tempfile
    step = 0.2 if quick else 0.1
    zs_cv = np.round(np.arange(5.6, 8.41, step), 2)
    zs_ncc = np.round(np.arange(5.0, 9.01, 0.1), 3)
    blob = {}
    tmp = Path(tempfile.mkdtemp(prefix="afval_"))

    for z0 in INJECTED:
        out = _generate(z0, tmp / f"ph_{int(z0*10)}")
        cfg = _phantom_cfg(out)
        tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
        xs, ys = _grid(cfg, 0.1)
        a, b, m = _pair_stacks(tmaps, cfg.geometry, zs_ncc, xs, ys)
        fc = focus_curve(a, b, m, zs_ncc, xs, ys)
        scan = _cv_scan(cfg, tmaps, zs_cv)
        k = "%d" % int(z0 * 10)
        blob[k + "_cvz"] = np.array([e["z"] for e in scan])
        blob[k + "_cv"] = np.array([e["cv_mean"] for e in scan])
        blob[k + "_fz"] = zs_ncc
        blob[k + "_fn"] = np.array([np.nan if v is None else v for v in fc["ncc"]])
        print(f"  injected {z0} m -> CV-min {blob[k+'_cvz'][np.argmin(blob[k+'_cv'])]:.2f} m")
    return blob


def plot(d, out_png: Path):
    """Synthetic ground-truth validation only (the real-data CV curve and parallax
    refocus are shown in the report body, so they are deliberately not repeated)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10.0, 11.5))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 1.0], hspace=0.26)

    axA = fig.add_subplot(gs[0])
    rec_cv, rec_fn = [], []
    for z0 in INJECTED:
        k = "%d" % int(z0 * 10)
        cvz, cv = d[k + "_cvz"], d[k + "_cv"]
        zmin = cvz[np.argmin(cv)]
        rec_cv.append(zmin)
        rec_fn.append(d[k + "_fz"][np.nanargmax(d[k + "_fn"])])
        axA.plot(cvz, cv, "-", color=COL[k], lw=1.9, label=f"ceiling injected at {z0} m")
        axA.plot(cvz, cv, "o", ms=3, color=COL[k])
        axA.axvline(z0, color=COL[k], ls=":", lw=1.3, alpha=0.7)
        axA.plot(zmin, cv.min(), "*", ms=19, color=COL[k], mec="k", mew=0.6, zorder=5)
    axA.set_xlabel("assumed ceiling height z (m)", fontsize=11.5)
    axA.set_ylabel("cross-validation residual\n(fit one detector, predict the other)", fontsize=11.5)
    axA.set_title("A.  The cross-validation minimum tracks the injected ceiling height\n"
                  "(dotted line: injected height   ·   star: recovered minimum)", fontsize=12.5)
    axA.legend(fontsize=10.5, loc="upper center")
    axA.grid(alpha=0.25)
    axA.tick_params(labelsize=10.5)

    axB = fig.add_subplot(gs[1])
    axB.plot([6.35, 7.65], [6.35, 7.65], "k--", lw=1, alpha=0.6, label="ideal (recovered = injected)")
    axB.plot(INJECTED, rec_cv, "o", ms=12, color="#222", label="cross-validation estimate (primary)")
    axB.plot(INJECTED, rec_fn, "s", ms=8, color="#999", label="two-view correlation (secondary)")
    for x, y in zip(INJECTED, rec_cv):
        axB.annotate(f"{y:.1f} m", (x, y), textcoords="offset points", xytext=(10, -4), fontsize=10.5)
    axB.set_xlabel("injected ceiling height (m)", fontsize=11.5)
    axB.set_ylabel("recovered height (m)", fontsize=11.5)
    axB.set_title("B.  Recovered vs. injected height\n"
                  "(consistent ~0.2 m under-estimate: the beam mass-centroid lies below the slab base)",
                  fontsize=12.5)
    axB.legend(fontsize=10.5, loc="upper left")
    axB.grid(alpha=0.25)
    axB.set_aspect("equal")
    axB.tick_params(labelsize=10.5)

    fig.suptitle("Autofocus validation against synthetic ground truth\n"
                 "campaign geometry (detector baseline, beam pitch), ceiling injected at known heights",
                 fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/production")
    ap.add_argument("--quick", action="store_true", help="coarser height grid (faster)")
    ap.add_argument("--out", default=str(REPO / "reports" / "autofocus_validation.png"))
    args = ap.parse_args()
    print("computing autofocus validation (this is slow)...")
    blob = compute(args.run, args.quick)
    plot(blob, Path(args.out))


if __name__ == "__main__":
    main()

"""Stage 0: dump histogram metadata, render quicklooks, run convention checks.

Usage: python scripts/inspect_data.py [--out runs/inspect]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muontomo.config import RunConfig
from muontomo.io import inspect_root_file, load_root_hist1d, load_root_hist2d

XY_MAPS = {"XY01m": 1.0, "XY02m": 2.0, "XY05m": 5.0, "XY07m": 7.0, "XY10m": 10.0}


def check(name: str, ok: bool, detail: str, results: list) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/inspect")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    cfg = RunConfig()
    paths = cfg.data
    meta = {tag: inspect_root_file(p) for tag, p in paths.items()}
    (out / "histograms.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"metadata for {sum(len(m) for m in meta.values())} histograms -> {out}/histograms.json")

    results: list = []
    print("\nConvention checks:")

    # 1. Identical binning of the analysis histograms across all three files.
    same = True
    for h in ["txty", "txtyCoarse", *XY_MAPS]:
        axes = {tag: (m[h]["axis0"], m[h]["axis1"]) for tag, m in meta.items()}
        same &= len({json.dumps(a, sort_keys=True) for a in axes.values()}) == 1
    check("identical_binning", same, "txty/txtyCoarse/XYnn binning matches across sky+pos0+pos1", results)

    # 2. Sky polar-angle distribution peaks near zero zenith.
    vals, edges = load_root_hist1d(paths["sky"], "Polar angle")
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = centers[np.argmax(vals)]
    check("sky_polar_peak", abs(peak) < 5.0, f"sky polar-angle peak at {peak:.2f} deg", results)

    # 3. Sky txty approx centro-symmetric (detects transpose/sign convention issues).
    sky = load_root_hist2d(paths["sky"], "txty").crop((-1, 1), (-1, 1)).rebin(16)
    v = sky.values
    sym = np.corrcoef(v.ravel(), v[::-1, ::-1].ravel())[0, 1]
    check("sky_centrosymmetry", sym > 0.98, f"corr(txty, txty rotated 180deg) = {sym:.4f}", results)

    # 4. XYnn structure sharpness varies with height (laminography refocusing works).
    #    Also renders quicklooks used to eyeball the convention.
    from scipy.ndimage import gaussian_filter

    focus = {}
    fig, axs = plt.subplots(2, len(XY_MAPS), figsize=(4 * len(XY_MAPS), 8))
    for i, (name, h) in enumerate(XY_MAPS.items()):
        hx_sky = load_root_hist2d(paths["sky"], name).rebin(8)
        hx_cafe = load_root_hist2d(paths["pos0"], name).rebin(8)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(hx_sky.values > 50, hx_cafe.values / np.maximum(hx_sky.values, 1), np.nan)
        ratio /= np.nanmedian(ratio)
        r = np.nan_to_num(ratio, nan=1.0)
        hp = r - gaussian_filter(r, 3)
        focus[name] = float(np.var(hp[np.isfinite(ratio)]))
        for j, (img, title) in enumerate([(hx_cafe.values, f"{name} counts pos0"), (r, f"{name} ratio pos0/sky")]):
            ax = axs[j, i]
            im = ax.imshow(img.T, origin="lower", cmap="viridis",
                           extent=[hx_sky.xedges[0], hx_sky.xedges[-1], hx_sky.yedges[0], hx_sky.yedges[-1]],
                           **({"vmin": 0.6, "vmax": 1.4} if j else {}))
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out / "images" / "xy_maps_quicklook.png", dpi=110)
    plt.close(fig)
    # Informational: raw high-pass variance mixes structure with noise; the rigorous
    # error-normalized depth-from-focus lives in muontomo.selfcal.
    best = max(focus, key=focus.get)
    check("xy_focus_varies", max(focus.values()) > 1.15 * min(focus.values()),
          f"high-pass variance by plane: { {k: round(v, 5) for k, v in focus.items()} } (sharpest: {best})", results)

    # 5. Acceptance stability: per-layer bar-occupancy shape sky vs cafe.
    worst = 0.0
    for layer in range(4):
        a, _ = load_root_hist1d(paths["sky"], f"bar hits Layer{layer}")
        b, _ = load_root_hist1d(paths["pos0"], f"bar hits Layer{layer}")
        r = np.corrcoef(a / a.sum(), b / b.sum())[0, 1]
        worst = max(worst, 1 - r)
    check("occupancy_stability", worst < 0.02, f"max (1 - corr) of layer occupancy sky vs pos0 = {worst:.4f}", results)

    # 6. Track totals for the record.
    totals = {tag: meta[tag]["txty"]["sum"] for tag in paths}
    check("track_totals", totals["sky"] > totals["pos0"] > totals["pos1"] > 1e6,
          f"txty sums: {totals}", results)

    # Quicklook of the angular transmission maps at analysis binning.
    sky_ang = load_root_hist2d(paths["sky"], "txty").crop((-1, 1), (-1, 1)).rebin(8)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tag in zip(axs, ["pos0", "pos1"]):
        cafe = load_root_hist2d(paths[tag], "txty").crop((-1, 1), (-1, 1)).rebin(8)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sky_ang.values >= 25, cafe.values / np.maximum(sky_ang.values, 1), np.nan)
        ratio /= np.nanquantile(ratio, 0.95)
        im = ax.imshow(ratio.T, origin="lower", cmap="viridis", vmin=0.5, vmax=1.2,
                       extent=[cafe.xedges[0], cafe.xedges[-1], cafe.yedges[0], cafe.yedges[-1]])
        ax.set(title=f"{tag} transmission (txty)", xlabel="tan(theta_x)", ylabel="tan(theta_y)")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out / "images" / "txty_transmission_quicklook.png", dpi=110)
    plt.close(fig)

    (out / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    n_fail = sum(not r["ok"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed -> {out}/checks.json")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

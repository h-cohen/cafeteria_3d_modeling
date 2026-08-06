"""Build a self-contained viewer.html for a run: no external requests, works
from file:// or as a private artifact.

    python -m muontomo.viewer.build --run runs/exp01
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from ..export import export_volume

VENDOR = Path(__file__).parent / "vendor"
MAX_DIM = 128  # cap embedded volume size; downsample larger grids


def _downsample(rho: np.ndarray, spacing: float, origin) -> tuple[np.ndarray, float, tuple]:
    factor = max(1, int(np.ceil(max(rho.shape) / MAX_DIM)))
    if factor == 1:
        return rho, spacing, origin
    from scipy.ndimage import zoom

    rho2 = zoom(rho, 1.0 / factor, order=1)
    return rho2.astype(np.float32), spacing * factor, origin


def _crop_xy(rho: np.ndarray, spacing: float, origin, box) -> tuple[np.ndarray, tuple]:
    (x0, x1), (y0, y1) = box
    i0 = max(0, int(np.floor((x0 - origin[0]) / spacing)))
    i1 = min(rho.shape[0], int(np.ceil((x1 - origin[0]) / spacing)))
    j0 = max(0, int(np.floor((y0 - origin[1]) / spacing)))
    j1 = min(rho.shape[1], int(np.ceil((y1 - origin[1]) / spacing)))
    cropped = rho[i0:i1, j0:j1, :]
    new_origin = (origin[0] + i0 * spacing, origin[1] + j0 * spacing, origin[2])
    return cropped, new_origin


def build_viewer(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    run = Path(run_dir)
    export_volume(run)  # writes volume.npy + meta.json (headline metrics, iso hints)
    rho = np.load(run / "volume.npy").astype(np.float32)
    meta = json.loads((run / "meta.json").read_text())

    pos = np.maximum(rho, 0.0)
    origin = tuple(meta["origin_m"])
    spacing = meta["spacing_m"]
    # The reconstruction grid is deliberately wider than the well-covered region
    # (limited-angle SIRT/TV pools spurious mass at whichever boundary is
    # worst-constrained, so keeping slack room around the real structure keeps
    # that bias off it) -- crop to the sub-box the config marks as trustworthy
    # before it ever reaches the viewer.
    crop = meta.get("viewer_crop_xy_m")
    if crop is not None:
        pos, origin = _crop_xy(pos, spacing, origin, crop)

    pos, spacing, origin = _downsample(pos, spacing, origin)

    # Load the run config + calibrated data once: they set the honest display
    # resolution (below) and provide the optional measured-data terrain layer.
    # Optional: everything degrades gracefully if the data files are absent.
    cfg = tmaps = None
    try:
        from ..calibration import transmission_maps
        from ..config import RunConfig
        from ..io import load_dataset

        cfg = RunConfig.load(run / "config.json")
        tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    except Exception as e:  # data files absent (e.g. moved run dir) -- viewer still works
        print(f"note: run data unavailable, no measured-data layer embedded ({e})")

    if cfg is not None and cfg.reconstruction.layered_zs:
        z_layer = float(cfg.reconstruction.layered_zs[0])
    else:
        zprof = pos.sum(axis=(0, 1))
        z_layer = origin[2] + int(np.argmax(zprof)) * spacing

    # The solver runs at voxels much finer than the data resolves (necessary to
    # avoid aliasing the beam periodicity). The TV-regularized solve already
    # suppresses sub-beam noise, so the DISPLAY needs only a light xy blur to
    # smooth residual voxel-scale speckle -- HALF the angular-bin footprint
    # (bin width x lever arm). Blurring by the full footprint was verified to
    # over-fatten the beams into wavy blobs; at half of it the beams read as
    # thin straight lines matching the single-detector 2D image, while the
    # inter-beam background stays clean (z axis untouched -- keeps layer height).
    from scipy.ndimage import gaussian_filter

    sigma_m = 0.12  # fallback when the data (and so the bin width) is unavailable
    if tmaps is not None:
        tm0 = next(iter(tmaps.values()))
        bin_w = float(tm0.txedges[1] - tm0.txedges[0])
        lever = z_layer - float(np.mean([cfg.geometry.pose(p).z for p in tmaps]))
        if bin_w * lever > 0:
            sigma_m = 0.5 * bin_w * lever
    pos = gaussian_filter(pos, sigma=(sigma_m / spacing, sigma_m / spacing, 0.0))
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0
    scale = 255.0 / max(p99 * 1.5, 1e-9)
    quant = np.clip(pos * scale, 0, 255).astype(np.uint8)
    data_b64 = base64.b64encode(quant.tobytes()).decode("ascii")

    # Model-free "measured data" surface: the mean per-detector backprojection of
    # -ln(T) onto the ceiling plane, sampled on the same xy grid as the volume.
    # This is the closest-to-raw-data view (no solver, no regularization) and is
    # offered in the viewer as an alternative terrain source, so the
    # reconstruction can always be sanity-checked against what the detectors
    # actually saw.
    data_layer_b64 = None
    data_scale = None
    if tmaps is not None:
        from ..backproject import backproject_opacity

        xs = origin[0] + (np.arange(pos.shape[0]) + 0.5) * spacing
        ys = origin[1] + (np.arange(pos.shape[1]) + 0.5) * spacing
        _, mean_bp = backproject_opacity(tmaps, cfg.geometry, z_layer, xs, ys)
        mean_bp = np.nan_to_num(mean_bp, nan=0.0)
        bp99 = float(np.percentile(mean_bp[mean_bp > 0], 99)) if (mean_bp > 0).any() else 1.0
        data_scale = 255.0 / max(bp99 * 1.5, 1e-9)
        data_layer_b64 = base64.b64encode(
            np.clip(mean_bp * data_scale, 0, 255).astype(np.uint8).tobytes()
        ).decode("ascii")

    # DIP-enhanced surface: the Deep Image Prior reconstruction of the ceiling
    # layer (muontomo.enhance), the winner of the enhancement comparison -- best
    # beam-position accuracy (~0.04 m) and clean sharp beams, gated against the
    # measured data so it cannot invent structure. Prefer the precomputed
    # enhance/dip.npy (deterministic, already verified by the CLI); fall back to
    # computing it here. The enhance layer lives on the full solver grid, so it
    # is interpolated onto the same display grid as the other surfaces.
    dip_layer_b64 = None
    dip_scale = None
    if tmaps is not None:
        try:
            dfile = run / "enhance" / "dip.npy"
            if dfile.exists():
                dip_full = np.load(dfile).astype(np.float64)
            else:
                from ..enhance.context import load_context
                from ..enhance import dip as _dipmod

                dip_full = np.asarray(_dipmod._DIP().enhance(load_context(run)), np.float64)
            # interpolate the full-grid layer onto the display xy grid
            from scipy.interpolate import RegularGridInterpolator

            o0, s0 = meta["origin_m"], meta["spacing_m"]
            xs0 = o0[0] + (np.arange(dip_full.shape[0]) + 0.5) * s0
            ys0 = o0[1] + (np.arange(dip_full.shape[1]) + 0.5) * s0
            interp = RegularGridInterpolator((xs0, ys0), dip_full, bounds_error=False, fill_value=0.0)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            dip_disp = np.maximum(interp((gx, gy)), 0.0)
            d99 = float(np.percentile(dip_disp[dip_disp > 0], 99)) if (dip_disp > 0).any() else 1.0
            dip_scale = 255.0 / max(d99 * 1.5, 1e-9)
            dip_layer_b64 = base64.b64encode(
                np.clip(dip_disp * dip_scale, 0, 255).astype(np.uint8).tobytes()
            ).decode("ascii")
        except Exception as e:  # torch absent, etc. -- surface simply won't appear
            print(f"note: no DIP-enhanced layer embedded ({e})")

    # Verified beam positions (model-free, from muontomo.beams via the run's
    # metrics.json): drawn in the viewer as guide lines over the terrain, so the
    # beams the raw data proves are there can be located at a glance even where
    # the reconstruction renders them faintly.
    verified_beams = None
    mfile = run / "metrics.json"
    if mfile.exists():
        beams = json.loads(mfile.read_text()).get("beams") or {}
        if "beams_x_data_m" in beams:
            verified_beams = {
                "x": beams.get("beams_x_data_m", []),
                "y": beams.get("beams_y_data_m", []),
                "z": z_layer,
            }

    # Percentile-based iso levels on the actual (smoothed, cropped) field: p80
    # sits just above the noise floor, p92 isolates the strongest beam ridges.
    active = pos[pos > 0]
    if active.size:
        lo_val, hi_val = np.percentile(active, [80, 92])
    else:
        lo_val, hi_val = 0.1, 0.2
    suggested_iso = [
        round(float(np.clip(lo_val * scale, 1, 254)), 1),
        round(float(np.clip(hi_val * scale, 1, 254)), 1),
    ]

    volume_payload = {
        "shape": list(quant.shape),
        "origin_m": list(origin),
        "spacing_m": spacing,
        "value_range": meta["value_range"],
        "quant_scale": scale,  # physical density = byte_value / quant_scale
        "suggested_iso": suggested_iso,
        "run": meta["run"],
        "headline_metrics": meta.get("headline_metrics", {}),
        "detectors": meta.get("detectors", []),
        "data_b64": data_b64,
        "backproject_b64": data_layer_b64,  # nx*ny uint8, or null
        "backproject_scale": data_scale,  # physical opacity = byte / scale
        "dip_b64": dip_layer_b64,  # nx*ny uint8 DIP-enhanced layer, or null
        "dip_scale": dip_scale,  # physical density = byte / scale
        "verified_beams": verified_beams,  # {x: [..], y: [..], z} or null
    }

    html = (VENDOR.parent / "template.html").read_text()
    html = html.replace("/*__THREE_JS__*/", (VENDOR / "three.min.js").read_text())
    html = html.replace("/*__ORBIT_CONTROLS_JS__*/", (VENDOR / "OrbitControls.js").read_text())
    html = html.replace("/*__MARCHING_CUBES_JS__*/", (VENDOR / "marchingcubes.js").read_text())
    html = html.replace("/*__VOLUME_JSON__*/", json.dumps(volume_payload))
    html = html.replace("/*__APP_JS__*/", (VENDOR.parent / "app.js").read_text())

    out = Path(out_path) if out_path else run / "viewer.html"
    out.write_text(html)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = build_viewer(args.run, args.out)
    print(f"viewer -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

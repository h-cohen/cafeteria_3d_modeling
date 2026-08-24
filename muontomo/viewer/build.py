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


def _support_xy_taper(run, origin, spacing, shape, x_margin=0.8, y_edge=1.0, roll=0.7):
    """xy taper in [0,1] bounding the structure to the verified beams' x-extent
    (+margin) and the interior y, cosine-rolling to 0 over the grid rim. Applied to
    the full-3D volume so its bright limited-angle boundary ring is removed and the
    faint interior beams survive the display normalization. Same shape as the
    'clean' surface's coverage taper. Returns None if beam positions are unknown."""
    mfile = run / "metrics.json"
    if not mfile.exists():
        return None
    beams = (json.loads(mfile.read_text()).get("beams") or {}).get("beams_x_data_m")
    if not beams:
        return None
    bx = np.asarray(beams, float)
    nx, ny = shape[0], shape[1]
    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing

    def axis(coord, lo, hi):
        d = np.minimum(coord - lo, hi - coord)
        w = np.clip(1.0 + d / roll, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * w)

    wx = axis(xs, float(bx.min()) - x_margin, float(bx.max()) + x_margin)
    wy = axis(ys, origin[1] + y_edge, origin[1] + ny * spacing - y_edge)
    return np.outer(wx, wy).astype(np.float32)


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

    # Full-room 3D voxel reconstruction (optional): the same data solved over the
    # WHOLE grid (SIRT+TV) instead of the thin ceiling layer, dropped in as
    # volume_full3d.npz. Two views make this limited-angle (z-smeared), so it is
    # offered as a toggle -- swap the volume that drives the iso-surfaces and the
    # z-slice, to inspect the full voxel cloud rather than just the ceiling plane.
    # Run through the identical crop -> downsample as the main volume so it shares
    # the display grid (same nx, ny, nz), then a light isotropic smooth.
    volume_full_b64 = volume_full_scale = None
    ff = run / "volume_full3d.npz"
    if ff.exists():
        try:
            with np.load(ff) as vz:
                rf = np.maximum(vz["rho"].astype(np.float32), 0.0)
            o2, s2 = tuple(meta["origin_m"]), meta["spacing_m"]
            if crop is not None:
                rf, o2 = _crop_xy(rf, s2, o2, crop)
            rf, s2, o2 = _downsample(rf, s2, o2)
            rf = gaussian_filter(rf, (sigma_m / s2, sigma_m / s2, 0.6))
            # strip the limited-angle boundary ring so the display normalization is
            # set by the interior structure, not the artifacts (see _support_xy_taper)
            taper = _support_xy_taper(run, o2, s2, rf.shape)
            if taper is not None:
                rf = rf * taper[:, :, None]
            if rf.shape == pos.shape:
                f99 = float(np.percentile(rf, 99)) if rf.max() > 0 else 1.0
                volume_full_scale = 255.0 / max(f99 * 1.5, 1e-9)
                volume_full_b64 = base64.b64encode(
                    np.clip(rf * volume_full_scale, 0, 255).astype(np.uint8).tobytes()
                ).decode("ascii")
            else:
                print(f"note: full-3D volume shape {rf.shape} != display {pos.shape}; skipped")
        except Exception as e:
            print(f"note: no full-3D volume embedded ({e})")

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

    # Enhanced surfaces (muontomo.enhance), each a 2D ceiling layer on the full
    # solver grid, interpolated onto the display grid like every other surface and
    # gated against the measured data so they cannot invent structure:
    #   dip   -- Deep Image Prior, best beam-position accuracy (~0.04 m).
    #   clean -- artifact removal + sharpening (coverage taper kills the boundary
    #            blobs, guided denoise + floor flatten the background, direction-aware
    #            sharpen tightens the beams). ~6x beam contrast-to-noise vs the raw slice.
    # Prefer the precomputed enhance/<name>.npy (deterministic, CLI-verified).
    def _embed_enhance(name, fallback=None):
        try:
            f = run / "enhance" / f"{name}.npy"
            if f.exists():
                full = np.load(f).astype(np.float64)
            elif fallback is not None:
                full = np.asarray(fallback(), np.float64)
            else:
                return None, None
            from scipy.interpolate import RegularGridInterpolator

            o0, s0 = meta["origin_m"], meta["spacing_m"]
            xs0 = o0[0] + (np.arange(full.shape[0]) + 0.5) * s0
            ys0 = o0[1] + (np.arange(full.shape[1]) + 0.5) * s0
            interp = RegularGridInterpolator((xs0, ys0), full, bounds_error=False, fill_value=0.0)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            disp = np.maximum(interp((gx, gy)), 0.0)
            p99 = float(np.percentile(disp[disp > 0], 99)) if (disp > 0).any() else 1.0
            sc = 255.0 / max(p99 * 1.5, 1e-9)
            b64 = base64.b64encode(
                np.clip(disp * sc, 0, 255).astype(np.uint8).tobytes()
            ).decode("ascii")
            return b64, sc
        except Exception as e:  # torch absent / layer missing -- surface just won't appear
            print(f"note: no {name}-enhanced layer embedded ({e})")
            return None, None

    dip_layer_b64 = dip_scale = None
    clean_layer_b64 = clean_scale = None
    dipclean_layer_b64 = dipclean_scale = None
    if tmaps is not None:
        def _dip_fallback():
            from ..enhance.context import load_context
            from ..enhance import dip as _dipmod

            return _dipmod._DIP().enhance(load_context(run))

        dip_layer_b64, dip_scale = _embed_enhance("dip", _dip_fallback)
        clean_layer_b64, clean_scale = _embed_enhance("clean")
        dipclean_layer_b64, dipclean_scale = _embed_enhance("dipclean")

    # Single-detector reconstructions: the same SIRT+TV solve trained on ONE
    # position only (volume_holdout_pos0/pos1.npz, written by reconstruct.py's
    # cross-validation pass). Embedded as 2D layers at the fitted ceiling height,
    # run through the identical crop -> downsample -> display-blur pipeline as the
    # both-detector volume, so the viewer can toggle between "one detector" and
    # "two detectors fused" and see directly what the second view buys (the
    # limited-angle streaking a single detector leaves behind).
    def _single_detector_layer(vol_path: Path):
        if not vol_path.exists():
            return None, None
        r = np.maximum(np.load(vol_path)["rho"].astype(np.float32), 0.0)
        o0, s0 = tuple(meta["origin_m"]), meta["spacing_m"]
        if crop is not None:
            r, o0 = _crop_xy(r, s0, o0, crop)
        r, s0, o0 = _downsample(r, s0, o0)
        r = gaussian_filter(r, sigma=(sigma_m / s0, sigma_m / s0, 0.0))
        iz = max(0, min(r.shape[2] - 1, int(round((z_layer - o0[2]) / s0))))
        lyr = r[:, :, iz]
        v99 = float(np.percentile(lyr[lyr > 0], 99)) if (lyr > 0).any() else 1.0
        sc = 255.0 / max(v99 * 1.5, 1e-9)
        b64 = base64.b64encode(
            np.clip(lyr * sc, 0, 255).astype(np.uint8).tobytes()
        ).decode("ascii")
        return b64, sc

    pos0_b64, pos0_scale = _single_detector_layer(run / "volume_holdout_pos0.npz")
    pos1_b64, pos1_scale = _single_detector_layer(run / "volume_holdout_pos1.npz")

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

    # Model-free autofocus (muontomo.focus): the ceiling height the two-view
    # parallax itself points to, independent of the solver's assumed layer. Shown
    # in the HUD next to the solve height so the data-chosen depth is visible at a
    # glance. Prefer the precomputed focus.json; else run a coarse global scan.
    # NOTE: this is a measurement, not the display plane -- the volumes were
    # SOLVED at z_layer, so the terrain still slices there; z* feeds a re-solve.
    focus_info = None
    if tmaps is not None:
        try:
            ffile = run / "focus.json"
            if ffile.exists():  # authoritative: cross-validated height from evaluate
                fj = json.loads(ffile.read_text())
                focus_info = {
                    "z_m": fj.get("autofocus_z_m"),  # cross-validated (trusted)
                    "quicklook_z_m": fj.get("quicklook_z_m"),  # model-free NCC
                    "map_median_z_m": (fj.get("height_map") or {}).get("median_z_m"),
                    "map_iqr_z_m": (fj.get("height_map") or {}).get("iqr_z_m"),
                    "solve_z_m": fj.get("solve_z_m", round(float(z_layer), 2)),
                }
            else:  # fast fallback: model-free quick-look only (no CV scan at build time)
                from ..focus import quick_focus

                fj = quick_focus(tmaps, cfg.geometry, cfg)
                focus_info = {
                    "z_m": None,
                    "quicklook_z_m": fj.get("focus_z_m"),
                    "solve_z_m": round(float(z_layer), 2),
                }
        except Exception as e:  # data absent, etc. -- HUD line simply won't appear
            print(f"note: no autofocus height embedded ({e})")

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
        "volume_full_b64": volume_full_b64,  # nx*ny*nz uint8 full-room 3D volume, or null
        "volume_full_scale": volume_full_scale,  # physical density = byte / scale
        "backproject_b64": data_layer_b64,  # nx*ny uint8, or null
        "backproject_scale": data_scale,  # physical opacity = byte / scale
        "dip_b64": dip_layer_b64,  # nx*ny uint8 DIP-enhanced layer, or null
        "dip_scale": dip_scale,  # physical density = byte / scale
        "clean_b64": clean_layer_b64,  # nx*ny uint8 artifact-cleaned layer, or null
        "clean_scale": clean_scale,  # physical density = byte / scale
        "dipclean_b64": dipclean_layer_b64,  # nx*ny uint8 DIP+cleaned layer, or null
        "dipclean_scale": dipclean_scale,  # physical density = byte / scale
        "pos0_b64": pos0_b64,  # nx*ny uint8 single-detector (pos0-only) layer, or null
        "pos0_scale": pos0_scale,  # physical density = byte / scale
        "pos1_b64": pos1_b64,  # nx*ny uint8 single-detector (pos1-only) layer, or null
        "pos1_scale": pos1_scale,  # physical density = byte / scale
        "focus": focus_info,  # {z_m, ncc, map_median_z_m, map_iqr_z_m, solve_z_m} or null
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

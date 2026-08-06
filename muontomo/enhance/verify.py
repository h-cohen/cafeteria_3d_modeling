"""Shared anti-hallucination gate + sharpness metrics for an enhanced layer.

The reference beam positions come from the measured data itself (peaks of the
backprojection guide's central-band x-profile) -- so "did the enhancer move or
invent beams?" is answered against real data, not against another reconstruction.
"""

from __future__ import annotations

import json

import numpy as np

from ..beams import beam_peaks
from .context import EnhanceContext

# Gate thresholds: the plain reconstruction currently sits at ~0.10 m mean beam
# offset with exactly 5 beams; an enhancer must not do worse on either.
MAX_MEAN_OFFSET_M = 0.15
EXPECT_N_BEAMS = 5


def _central_x_profile(ctx: EnhanceContext, img: np.ndarray) -> np.ndarray:
    band = (ctx.ys > -2.0) & (ctx.ys < 2.0)
    return np.asarray(img)[:, band].mean(axis=1)


def _well_covered(ctx: EnhanceContext) -> np.ndarray:
    """x-mask dropping the low-coverage grid margins where edge artifacts live
    (1 m in from each grid edge, matching muontomo.beams)."""
    (x0, x1), _ = ctx.cfg.geometry.grid_xy_m
    return (ctx.xs > x0 + 1.0) & (ctx.xs < x1 - 1.0)


def data_beam_positions(ctx: EnhanceContext) -> np.ndarray:
    """Authoritative model-free beam x-positions: the parallax-verified values in
    the run's metrics.json (muontomo.beams). Falls back to peaks of the guide's
    central profile over the well-covered window if metrics are absent."""
    mfile = ctx.run / "metrics.json"
    if mfile.exists():
        beams = (json.loads(mfile.read_text()).get("beams") or {}).get("beams_x_data_m")
        if beams:
            return np.asarray(beams, float)
    m = _well_covered(ctx)
    return beam_peaks(ctx.xs[m], _central_x_profile(ctx, ctx.guide)[m])


def _beam_fwhm_m(ctx: EnhanceContext, prof: np.ndarray, peaks_x: np.ndarray) -> float:
    """Mean full-width-half-max of the beams in world metres (sharpness proxy)."""
    from scipy.signal import find_peaks, peak_widths

    ok = np.isfinite(prof)
    xs, p = ctx.xs[ok], prof[ok]
    if p.size < 5 or peaks_x.size == 0:
        return float("nan")
    prom = 0.5 * float(np.nanstd(p))
    idx, _ = find_peaks(p, prominence=prom)
    if idx.size == 0:
        return float("nan")
    widths = peak_widths(p, idx, rel_height=0.5)[0]  # in samples
    return float(np.median(widths) * ctx.spacing)


def _edge_gradient(ctx: EnhanceContext, img: np.ndarray) -> float:
    """Mean in-plane gradient magnitude, normalised -- higher = crisper edges."""
    gx, gy = np.gradient(np.asarray(img, float))
    g = np.hypot(gx, gy)
    scale = np.percentile(img[img > 0], 90) if (img > 0).any() else 1.0
    return float(g.mean() / (scale + 1e-12) / ctx.spacing)


def verify(ctx: EnhanceContext, enhanced: np.ndarray, runtime_s: float | None = None,
           extra: dict | None = None) -> dict:
    """Beam-accuracy gate + sharpness metrics for an enhanced 2D layer."""
    ref = data_beam_positions(ctx)
    prof = _central_x_profile(ctx, enhanced)
    core = _well_covered(ctx)
    peaks = beam_peaks(ctx.xs[core], prof[core])
    offsets = [float(peaks[np.argmin(np.abs(peaks - r))] - r) for r in ref] if peaks.size else []
    mean_off = float(np.mean(np.abs(offsets))) if offsets else float("nan")
    n_beams = int(peaks.size)
    passed = (
        peaks.size > 0
        and mean_off <= MAX_MEAN_OFFSET_M
        and abs(n_beams - EXPECT_N_BEAMS) <= 1
    )
    out = {
        "mean_abs_beam_offset_m": round(mean_off, 3) if offsets else None,
        "n_beams": n_beams,
        "n_beams_data": int(ref.size),
        "beam_positions_m": [round(float(p), 2) for p in peaks],
        "beam_fwhm_m": round(_beam_fwhm_m(ctx, prof, peaks), 3),
        "edge_gradient": round(_edge_gradient(ctx, enhanced), 4),
        "verdict": "PASS" if passed else "FAIL",
    }
    if runtime_s is not None:
        out["runtime_s"] = round(runtime_s, 2)
    if extra:
        out.update(extra)
    return out

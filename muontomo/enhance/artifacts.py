"""Artifact removal + sharpening of the reconstruction layer (scipy-only).

Composes three prior-guided, anti-hallucination-safe stages into one cleaned
surface:

  Stage 1 -- COVERAGE MASK.  Limited-angle SIRT/TV pools spurious mass at the
    grid boundary, where voxels are constrained by only a few grazing rays. We
    count, per detector, how many rays actually touch each ceiling pixel (from
    the thin-layer forward model) and taper the density to zero where either
    detector's coverage is thin. This deletes the bright corner/edge blobs
    without inventing anything -- it only suppresses regions the data cannot
    constrain.

  Stage 2 -- FLOOR SUPPRESSION + EDGE-PRESERVING DENOISE.  A guided filter
    (edges taken from the cleanest single-detector backprojection) removes the
    grainy inter-beam speckle while keeping beam edges; the residual DC pedestal
    between the beams is then estimated away from the verified beam positions and
    subtracted, flattening the background toward zero.

  Stage 3 -- DIRECTION-AWARE SHARPEN.  Using the beams' known straight geometry,
    smooth ALONG each ridge (via a structure-orientation gate, so the vertical
    beams straighten but the horizontal cross-beam is left alone) and unsharp
    ACROSS, narrowing the beams without amplifying isotropic noise.

Operates on the reconstruction slice (ctx.layer); no torch. Gated by the shared
beam verification like every other enhancer.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .base import register
from .context import EnhanceContext
from .guided import guided_filter

# --- tunables (conservative; keep all five beams through the gate) ---
X_MARGIN_M = 0.8       # keep the support this far beyond the outermost verified beams
Y_EDGE_M = 1.0         # start tapering y this far in from the grid edge
TAPER_M = 0.7          # cosine roll width of the support taper (m)
GUIDE_RADIUS_M = 0.4   # guided-filter window (beam scale)
GUIDE_EPS_FRAC = 0.1   # guided-filter regularization, as a fraction of guide std
FLOOR_FRAC = 0.6       # fraction of the inter-beam pedestal (p40) to subtract
FLOOR_PCT = 40         # inter-beam pedestal percentile
ALONG_SIGMA_M = 0.30   # along-beam smoothing (straightening)
UNSHARP_SIGMA_M = 0.25 # across-beam unsharp scale
UNSHARP_AMOUNT = 0.6   # across-beam unsharp strength


def _axis_taper(coord: np.ndarray, lo_full: float, hi_full: float, roll: float) -> np.ndarray:
    """1.0 inside [lo_full, hi_full]; cosine-rolls to 0 over `roll` metres beyond each side."""
    d = np.minimum(coord - lo_full, hi_full - coord)  # +inside, - outside
    w = np.clip(1.0 + d / roll, 0.0, 1.0)             # 1 inside, ramps 1->0 over `roll` outside
    return 0.5 - 0.5 * np.cos(np.pi * w)


def coverage_mask(ctx: EnhanceContext) -> np.ndarray:
    """[nx, ny] structure-support taper in [0, 1]. The limited-angle solver pools
    spurious mass on the grid rim BEYOND the real room (measured here: ~2.7x
    brighter outside the room than inside), and ray-count coverage does not flag it
    (it is actually higher at the corners). Instead we bound the support by the
    verified beams' extent in x (data-driven) plus a margin, and taper the outer
    grid band in y, so the corner/edge blobs fall to zero while every beam is kept."""
    from .verify import data_beam_positions

    bx = data_beam_positions(ctx)
    (x0, x1), (y0, y1) = ctx.cfg.geometry.grid_xy_m
    wx = _axis_taper(ctx.xs, float(bx.min()) - X_MARGIN_M, float(bx.max()) + X_MARGIN_M, TAPER_M)
    wy = _axis_taper(ctx.ys, y0 + Y_EDGE_M, y1 - Y_EDGE_M, TAPER_M)
    return np.clip(np.outer(wx, wy), 0.0, 1.0)


def _floor_subtract(ctx: EnhanceContext, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the inter-beam pedestal: estimated in the central band, away from
    the verified beams and away from the masked rim."""
    from .verify import data_beam_positions

    beams = data_beam_positions(ctx)
    band = (ctx.ys > -3.0) & (ctx.ys < 3.0)
    far = np.ones(ctx.xs.size, bool)
    for b in beams:
        far &= np.abs(ctx.xs - b) > 0.4
    region = img[np.ix_(far, band)]
    rmask = mask[np.ix_(far, band)]
    vals = region[(rmask > 0.5) & (region > 0)]
    floor = FLOOR_FRAC * float(np.percentile(vals, FLOOR_PCT)) if vals.size else 0.0
    return np.maximum(img - floor, 0.0)


def _denoise_and_floor(ctx: EnhanceContext, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Guided edge-preserving denoise, then subtract the inter-beam pedestal."""
    # Mean backprojection as the guide (not the single sharpest detector): the Stage-1
    # coverage taper already removes the low-coverage rim that would otherwise let the
    # mean dilute a beam, and the mean sits on the parallax-verified beam positions, so
    # it transfers edges without the single-view lateral shift.
    g = ctx.guide * mask
    nz = g[g > 0]
    eps = (GUIDE_EPS_FRAC * float(np.std(nz))) ** 2 if nz.size else 1e-6
    radius_px = max(1, int(round(GUIDE_RADIUS_M / ctx.spacing)))
    den = np.maximum(guided_filter(img, g, radius_px, eps), 0.0)
    return _floor_subtract(ctx, den, mask)


def directional_sharpen(ctx: EnhanceContext, img: np.ndarray) -> np.ndarray:
    """Smooth along ridges (gated to vertical-beam orientation) + unsharp across."""
    sp = ctx.spacing
    gx, gy = np.gradient(gaussian_filter(img, 1.0))
    Jxx = gaussian_filter(gx * gx, 2.0)
    Jyy = gaussian_filter(gy * gy, 2.0)
    # gate ~ 1 where structure varies in x (vertical beams), ~0 for horizontal cross-beam
    gate = Jxx / (Jxx + Jyy + 1e-12)
    along = gaussian_filter1d(img, ALONG_SIGMA_M / sp, axis=1)  # smooth along y
    straight = gate * along + (1.0 - gate) * img
    blur = gaussian_filter(straight, UNSHARP_SIGMA_M / sp)
    out = straight + UNSHARP_AMOUNT * (straight - blur)
    return np.maximum(out, 0.0)


def _cnr(ctx: EnhanceContext, img: np.ndarray) -> tuple:
    """(contrast-to-noise ratio, background noise) of the beams vs the floor, in
    the central band -- the headline artifact/noise metrics."""
    from .verify import data_beam_positions

    band = (ctx.ys > -2.0) & (ctx.ys < 2.0)
    prof = np.asarray(img)[:, band].mean(axis=1)
    beams = data_beam_positions(ctx)
    near = np.zeros(ctx.xs.size, bool)
    for b in beams:
        near |= np.abs(ctx.xs - b) < 0.25
    (x0, x1), _ = ctx.cfg.geometry.grid_xy_m
    core = (ctx.xs > x0 + 1.0) & (ctx.xs < x1 - 1.0)
    fg = prof[near & core]
    bg = prof[(~near) & core]
    if fg.size < 2 or bg.size < 2:
        return float("nan"), float("nan")
    noise = float(np.std(bg))
    cnr = float((fg.mean() - bg.mean()) / (noise + 1e-12))
    scale = float(np.percentile(prof[core], 99)) or 1.0
    return cnr, noise / (scale + 1e-12)


class _Clean:
    name = "clean"

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        base = ctx.display_blur(ctx.layer)
        cnr0, fn0 = _cnr(ctx, base)

        mask = coverage_mask(ctx)                        # Stage 1
        x = base * mask
        x = _denoise_and_floor(ctx, x, mask)             # Stage 2
        x = directional_sharpen(ctx, x)                  # Stage 3
        x = np.maximum(x * mask, 0.0)

        cnr1, fn1 = _cnr(ctx, x)
        self.last_info = {
            "cnr_before": round(cnr0, 3), "cnr_after": round(cnr1, 3),
            "flat_noise_before": round(fn0, 4), "flat_noise_after": round(fn1, 4),
            "coverage_kept_frac": round(float((mask > 0.5).mean()), 3),
        }
        return x


class _DipClean:
    """Artifact cleanup applied on top of the Deep Image Prior layer: DIP has the
    best beam-position accuracy (~0.04 m) and is already data-consistently
    denoised and sharp, so this combo only removes what DIP cannot -- the
    limited-angle boundary blobs (Stage-1 support taper) and the residual
    inter-beam pedestal (floor subtraction). No guided re-denoise and no
    re-sharpening: those would only perturb DIP's verified beam positions.
    Needs torch (via the dip module) unless a precomputed enhance/dip.npy exists.
    """

    name = "dipclean"

    def _dip_layer(self, ctx: EnhanceContext) -> np.ndarray:
        f = ctx.run / "enhance" / "dip.npy"
        if f.exists():  # deterministic, already gate-verified by the CLI
            return np.load(f).astype(np.float64)
        from . import dip as _dip

        return np.asarray(_dip._DIP().enhance(ctx), np.float64)

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        base = self._dip_layer(ctx)
        cnr0, fn0 = _cnr(ctx, base)

        mask = coverage_mask(ctx)          # Stage 1: kill the boundary ring
        x = base * mask
        x = _floor_subtract(ctx, x, mask)  # Stage 2: flatten the inter-beam pedestal
        x = np.maximum(x * mask, 0.0)

        cnr1, fn1 = _cnr(ctx, x)
        self.last_info = {
            "cnr_before": round(cnr0, 3), "cnr_after": round(cnr1, 3),
            "flat_noise_before": round(fn0, 4), "flat_noise_after": round(fn1, 4),
            "coverage_kept_frac": round(float((mask > 0.5).mean()), 3),
        }
        return x


register(_Clean())
register(_DipClean())

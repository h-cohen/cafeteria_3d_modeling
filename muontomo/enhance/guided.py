"""Guided filter (He, Sun & Tang 2010): sharpen the reconstruction layer using
the clean measured backprojection as the guide.

Output takes its VALUES from the reconstruction and its EDGES from the measured
data. Anti-hallucination property: in a window where guide and recon are
uncorrelated, a -> 0, so structure present in only one of them does NOT transfer
-- only features in BOTH get sharpened. scipy-only, no training.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .base import FnEnhancer, register
from .context import EnhanceContext

# Defaults chosen by the sweep in dev (radius ~ beam scale, eps ~ guide noise):
# sharpest panel that still passes the beam gate.
DEFAULT_RADIUS_M = 0.4
DEFAULT_EPS_FRAC = 0.1


def guided_filter(p: np.ndarray, guide: np.ndarray, radius_px: int, eps: float) -> np.ndarray:
    """Classic O(N) box-filter guided filter. p = input, guide = reference."""
    p = np.asarray(p, float)
    I = np.asarray(guide, float)
    # fill any non-finite guide pixels with the local mean so edges aren't faked
    if not np.isfinite(I).all():
        finite = np.isfinite(I)
        filled = np.where(finite, I, 0.0)
        norm = uniform_filter(finite.astype(float), radius_px, mode="nearest")
        loc = uniform_filter(filled, radius_px, mode="nearest") / np.maximum(norm, 1e-9)
        I = np.where(finite, I, loc)

    def box(a):
        return uniform_filter(a, size=radius_px, mode="nearest")

    mI, mp = box(I), box(p)
    mII, mIp = box(I * I), box(I * p)
    varI = mII - mI * mI
    covIp = mIp - mI * mp
    a = covIp / (varI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def _enhance(ctx: EnhanceContext, radius_m: float = DEFAULT_RADIUS_M,
             eps_frac: float = DEFAULT_EPS_FRAC) -> np.ndarray:
    p = ctx.display_blur(ctx.layer)  # same 0.12 m pre-blur the viewer applies
    g = ctx.sharp_guide  # cleanest single detector -- keeps all beams (see context)
    nz = g[g > 0]
    eps = (eps_frac * float(np.std(nz))) ** 2 if nz.size else 1e-6
    radius_px = max(1, int(round(radius_m / ctx.spacing)))
    out = guided_filter(p, g, radius_px, eps)
    return np.maximum(out, 0.0)


register(FnEnhancer("guided", _enhance))

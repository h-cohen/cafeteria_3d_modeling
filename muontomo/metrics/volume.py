"""Volume plausibility metrics (no ground truth needed)."""

from __future__ import annotations

import numpy as np


def volume_stats(rho: np.ndarray, z_centers: np.ndarray) -> dict:
    """Negativity, vertical localization, density sanity of a reconstruction."""
    total = np.abs(rho).sum()
    neg_mass = np.abs(rho[rho < 0]).sum()
    pos = np.maximum(rho, 0.0)
    zprof = pos.sum(axis=(0, 1))
    zsum = zprof.sum()
    out = {
        "neg_mass_frac": float(neg_mass / total) if total > 0 else 0.0,
        "p50": float(np.percentile(rho, 50)),
        "p99": float(np.percentile(rho, 99)),
    }
    if zsum > 0:
        p = zprof / zsum
        dz = float(z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 1.0
        entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0))
        out["z_peak_m"] = float(z_centers[np.argmax(zprof)])
        out["z_eff_width_m"] = float(np.exp(entropy) * dz)  # thin slab -> small width
    else:
        out["z_peak_m"] = float("nan")
        out["z_eff_width_m"] = float("nan")
    return out

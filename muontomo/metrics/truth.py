"""Phantom-only metrics against the known synthetic ground truth."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def rmse_scaled(rec: np.ndarray, tru: np.ndarray) -> float:
    """min_a ||a*rec - tru|| / ||tru|| (absolute calibration is not the algorithm's job)."""
    denom = float(np.sum(rec * rec))
    a = float(np.sum(rec * tru)) / denom if denom > 0 else 0.0
    return float(np.linalg.norm(a * rec - tru) / max(np.linalg.norm(tru), 1e-30))


def ssim3d(rec: np.ndarray, tru: np.ndarray, win: int = 7) -> float:
    """Mean local SSIM with a uniform window; L = p99 of the truth."""
    L = float(np.percentile(tru, 99))
    if L <= 0:
        return 0.0
    denom = float(np.sum(rec * rec))
    a = float(np.sum(rec * tru)) / denom if denom > 0 else 0.0
    x = a * rec.astype(np.float64)
    y = tru.astype(np.float64)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mx, my = uniform_filter(x, win), uniform_filter(y, win)
    mxx, myy, mxy = uniform_filter(x * x, win), uniform_filter(y * y, win), uniform_filter(x * y, win)
    vx, vy = mxx - mx * mx, myy - my * my
    cov = mxy - mx * my
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(ssim.mean())


def iou_dense(rec: np.ndarray, tru: np.ndarray, level: float = 0.5) -> float:
    """IoU of the 'dense' masks, each thresholded at level * its own p99."""
    ra = rec > level * max(np.percentile(rec, 99), 1e-30)
    ta = tru > level * max(np.percentile(tru, 99), 1e-30)
    union = np.logical_or(ra, ta).sum()
    return float(np.logical_and(ra, ta).sum() / union) if union else 0.0


def z_error_m(rec: np.ndarray, tru: np.ndarray, z_centers: np.ndarray) -> float:
    zr = np.maximum(rec, 0).sum(axis=(0, 1))
    zt = np.maximum(tru, 0).sum(axis=(0, 1))
    if zr.sum() == 0 or zt.sum() == 0:
        return float("nan")
    return float(abs(z_centers[np.argmax(zr)] - z_centers[np.argmax(zt)]))

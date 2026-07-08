"""Structural image metrics: beam periodicity, stripe contrast, noise.

Advisory metrics. Always computed on the measured transmission maps too, so
scores can be read relative to what the data itself supports.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def periodicity(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """FFT peak prominence over its radial annulus + detected period/orientation.

    Returns {"snr": peak/median-of-annulus, "period_bins": 1/|k*|, "angle_deg": ...}.
    """
    a = np.array(img, dtype=float)
    if mask is not None:
        a = np.where(mask, a, np.nan)
    mean = np.nanmean(a)
    a = np.nan_to_num(a - mean, nan=0.0)
    ny, nx = a.shape
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    kr = np.hypot(ky, kx)
    search = (kr > 2.0 / max(ny, nx)) & (kr < 0.5)  # exclude DC and Nyquist edge
    if not search.any():
        return {"snr": 0.0, "period_bins": 0.0, "angle_deg": 0.0}
    idx = np.unravel_index(np.argmax(np.where(search, P, 0)), P.shape)
    k_star = kr[idx]
    peak = P[idx]
    annulus = search & (kr > 0.7 * k_star) & (kr < 1.3 * k_star)
    # exclude the peak neighborhood (and its mirror) from the annulus baseline
    d_peak = np.hypot(ky - ky[idx[0], 0], kx - kx[0, idx[1]])
    d_mirr = np.hypot(ky + ky[idx[0], 0], kx + kx[0, idx[1]])
    annulus &= (d_peak > 0.05) & (d_mirr > 0.05)
    base = np.median(P[annulus]) if annulus.any() else 1e-30
    return {
        "snr": float(peak / max(base, 1e-30)),
        "period_bins": float(1.0 / max(k_star, 1e-9)),
        "angle_deg": float(np.degrees(np.arctan2(ky[idx[0], 0], kx[0, idx[1]]))),
    }


def stripe_stats(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Otsu split into dense/open pixels -> Michelson contrast and CNR."""
    a = np.asarray(img, dtype=float)
    m = np.isfinite(a) if mask is None else (mask & np.isfinite(a))
    vals = a[m]
    if vals.size < 16 or np.ptp(vals) == 0:
        return {"contrast": 0.0, "cnr": 0.0}
    thr = _otsu(vals)
    lo, hi = vals[vals <= thr], vals[vals > thr]
    if lo.size < 4 or hi.size < 4:
        return {"contrast": 0.0, "cnr": 0.0}
    mu_lo, mu_hi = float(lo.mean()), float(hi.mean())
    denom = mu_hi + mu_lo
    contrast = (mu_hi - mu_lo) / denom if denom != 0 else 0.0
    cnr = (mu_hi - mu_lo) / np.sqrt(lo.var() + hi.var() + 1e-30)
    return {"contrast": float(contrast), "cnr": float(cnr)}


def flat_noise(img: np.ndarray, mask: np.ndarray | None = None, sigma: float = 2.0) -> float:
    """High-pass residual std: noise level relative to a smooth background."""
    a = np.nan_to_num(np.asarray(img, dtype=float))
    hp = a - gaussian_filter(a, sigma)
    m = np.ones(a.shape, bool) if mask is None else mask
    return float(hp[m].std()) if m.any() else 0.0


def _otsu(vals: np.ndarray, nbins: int = 128) -> float:
    hist, edges = np.histogram(vals, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.where(w0 > 0, m0 / np.maximum(w0, 1), 0)
    mu1 = np.where(w1 > 0, (m0[-1] - m0) / np.maximum(w1, 1), 0)
    var_between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.argmax(var_between)])

"""Data-fidelity metrics: does the forward-projected volume explain the counts?"""

from __future__ import annotations

import numpy as np


def chi2_ndof(lam_meas: np.ndarray, lam_pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gaussian chi2 per used bin in opacity space."""
    used = w > 0
    n = max(int(used.sum()), 1)
    return float(np.sum(w * (lam_meas - lam_pred) ** 2) / n)


def deviance_ndof(n_cafe: np.ndarray, mu: np.ndarray, mask: np.ndarray) -> float:
    """Poisson deviance per used bin in count space (exact at low counts).

    mu: predicted cafe counts = scale * n_sky * T_pred.
    """
    m = mask & (mu > 0)
    n = np.where(m, n_cafe, 0.0)
    mu = np.where(m, mu, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(n > 0, n * np.log(n / mu), 0.0)
    dev = 2.0 * np.sum(np.where(m, mu - n + term, 0.0))
    return float(dev / max(int(m.sum()), 1))


def chi2_aligned(
    lam_meas: np.ndarray, lam_pred: np.ndarray, w: np.ndarray, max_shift: int = 3
) -> tuple[float, tuple[int, int]]:
    """Minimum chi2 over integer 2D shifts of the prediction.

    A large (chi2 - chi2_aligned) gap says the geometry (pose/height) is off,
    not the algorithm: fix alignment before touching the solver.
    """
    best = (np.inf, (0, 0))
    for di in range(-max_shift, max_shift + 1):
        for dj in range(-max_shift, max_shift + 1):
            p = np.roll(lam_pred, (di, dj), axis=(0, 1))
            v = chi2_ndof(lam_meas, p, w)
            if v < best[0]:
                best = (v, (di, dj))
    return best

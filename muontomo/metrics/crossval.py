"""Generalization metrics: the overfitting detectors.

cv_*: a volume reconstructed from ONE position must predict the OTHER position's
opacity map (only a scalar offset free). binholdout_*: a volume reconstructed
from 80% of bins must predict the held-out 20%.
"""

from __future__ import annotations

import numpy as np


def _free_offset_resid(lam: np.ndarray, pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    resid = lam - pred
    sw = w.sum()
    if sw > 0:
        resid = resid - (w * resid).sum() / sw
    return resid


def heldout_chi2(lam: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted chi2/n on held-out bins, with a free scalar offset."""
    used = w > 0
    n = max(int(used.sum()), 1)
    resid = _free_offset_resid(lam, pred, w)
    return float(np.sum(w * resid**2) / n)


def heldout_pearson(lam: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation between measured and predicted opacity."""
    m = w > 0
    if m.sum() < 3:
        return 0.0
    a, b, ww = lam[m], pred[m], w[m]
    am = a - np.average(a, weights=ww)
    bm = b - np.average(b, weights=ww)
    denom = np.sqrt(np.average(am**2, weights=ww) * np.average(bm**2, weights=ww))
    if denom <= 0:
        return 0.0
    return float(np.average(am * bm, weights=ww) / denom)

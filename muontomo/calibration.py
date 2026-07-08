"""Sky-calibrated transmission maps with Poisson errors and validity masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import BinningConfig, CalibrationConfig
from .io import Dataset, Hist2D


@dataclass
class TransmissionMap:
    """Per angular bin: T = n_cafe / (s * n_sky), with leading-order Poisson sigma.

    Arrays are indexed [itx, ity]; edges are in tan(theta) units.
    n_cafe / n_sky keep the raw counts so count-space (Poisson) metrics are exact.
    """

    T: np.ndarray
    sigma_T: np.ndarray
    mask: np.ndarray  # True where the bin is usable
    txedges: np.ndarray
    tyedges: np.ndarray
    n_cafe: np.ndarray
    n_sky: np.ndarray
    scale: float  # s: expected cafe/sky count ratio for open sky
    pose_id: str = ""

    @property
    def txcenters(self) -> np.ndarray:
        return 0.5 * (self.txedges[:-1] + self.txedges[1:])

    @property
    def tycenters(self) -> np.ndarray:
        return 0.5 * (self.tyedges[:-1] + self.tyedges[1:])


def prepare_angular_hist(h: Hist2D, binning: BinningConfig) -> Hist2D:
    """Crop the fine txty histogram to the acceptance window and rebin for statistics."""
    t = binning.t_max
    h = h.crop((-t, t), (-t, t))
    return h.rebin(binning.rebin)


def estimate_scale(n_cafe: np.ndarray, n_sky: np.ndarray, cal: CalibrationConfig) -> float:
    """Normalization nuisance s: the cafe/sky count ratio that maps open-sky bins to T=1.

    Total-count normalization is biased low by the absorber itself, so instead map the
    `norm_quantile` of the smoothed raw ratio to 1: the least-absorbed bins are treated
    as (nearly) open sky. Refined later as a per-pose opacity offset in reconstruction.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_sky > 0, n_cafe / np.maximum(n_sky, 1), np.nan)
    ok = np.isfinite(ratio) & (n_sky >= cal.min_sky)
    if cal.norm_smooth_bins > 0:
        sm = gaussian_filter(np.nan_to_num(ratio), cal.norm_smooth_bins)
        norm = gaussian_filter(ok.astype(float), cal.norm_smooth_bins)
        with np.errstate(invalid="ignore"):
            ratio = np.where(norm > 0.5, sm / np.maximum(norm, 1e-9), np.nan)
        ok &= np.isfinite(ratio)
    if not ok.any():
        raise ValueError("no usable bins to estimate the transmission scale")
    return float(np.quantile(ratio[ok], cal.norm_quantile))


def compute_transmission(
    cafe: Hist2D, sky: Hist2D, cal: CalibrationConfig, pose_id: str = ""
) -> TransmissionMap:
    if cafe.values.shape != sky.values.shape:
        raise ValueError("cafe and sky histograms have different binning")
    n_cafe, n_sky = cafe.values, sky.values
    scale = estimate_scale(n_cafe, n_sky, cal)
    mask = (n_sky >= cal.min_sky) & (n_cafe >= cal.min_cafe)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(mask, n_cafe / (scale * np.maximum(n_sky, 1)), 0.0)
        sigma = np.where(mask, T * np.sqrt(1 / np.maximum(n_cafe, 1) + 1 / np.maximum(n_sky, 1)), np.inf)
    return TransmissionMap(
        T=T,
        sigma_T=sigma,
        mask=mask,
        txedges=cafe.xedges,
        tyedges=cafe.yedges,
        n_cafe=n_cafe,
        n_sky=n_sky,
        scale=scale,
        pose_id=pose_id,
    )


def transmission_maps(ds: Dataset, binning: BinningConfig, cal: CalibrationConfig) -> dict:
    """TransmissionMap per position, from a real or phantom Dataset."""
    sky = prepare_angular_hist(ds.hist("sky", binning.hist), binning)
    out = {}
    for pid in ds.positions:
        cafe = prepare_angular_hist(ds.hist(pid, binning.hist), binning)
        out[pid] = compute_transmission(cafe, sky, cal, pose_id=pid)
    return out

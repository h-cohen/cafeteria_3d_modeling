"""Opacity maps: lambda = -ln T with propagated errors and least-squares weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import TransmissionMap

T_FLOOR = 1e-3  # clip only the lower tail; T > 1 is kept so lambda noise stays zero-mean


@dataclass
class OpacityMap:
    lam: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    txedges: np.ndarray
    tyedges: np.ndarray
    pose_id: str = ""

    @property
    def weights(self) -> np.ndarray:
        """1/sigma^2 on masked bins, 0 elsewhere."""
        w = np.zeros_like(self.lam)
        w[self.mask] = 1.0 / self.sigma[self.mask] ** 2
        return w


def opacity_map(tmap: TransmissionMap) -> OpacityMap:
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(tmap.mask, -np.log(np.clip(tmap.T, T_FLOOR, None)), 0.0)
        sigma = np.where(tmap.mask, tmap.sigma_T / np.clip(tmap.T, T_FLOOR, None), np.inf)
    return OpacityMap(
        lam=lam,
        sigma=sigma,
        mask=tmap.mask.copy(),
        txedges=tmap.txedges,
        tyedges=tmap.tyedges,
        pose_id=tmap.pose_id,
    )

"""The one forward model shared by reconstruction, evaluation, and phantom generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import GeometryConfig
from .geometry import VoxelGrid, auto_grid
from .raycast import build_system_matrix


@dataclass
class ForwardModel:
    """System matrix plus the bookkeeping to go volume <-> per-pose angular maps."""

    A: sparse.csr_matrix
    grid: VoxelGrid
    pose_ids: list
    txedges: np.ndarray
    tyedges: np.ndarray

    @property
    def map_shape(self) -> tuple[int, int]:
        return len(self.txedges) - 1, len(self.tyedges) - 1

    @property
    def n_bins(self) -> int:
        ntx, nty = self.map_shape
        return ntx * nty

    def rows(self, pose_id: str) -> slice:
        i = self.pose_ids.index(pose_id)
        return slice(i * self.n_bins, (i + 1) * self.n_bins)

    def predict_opacity(self, x: np.ndarray, offsets: dict | None = None) -> dict:
        """Volume (flat or 3D) -> {pose_id: lambda map [ntx, nty]}; offsets add c_pose."""
        y = self.A @ np.asarray(x).ravel()
        out = {}
        for pid in self.pose_ids:
            lam = y[self.rows(pid)].reshape(self.map_shape)
            out[pid] = lam + (offsets or {}).get(pid, 0.0)
        return out

    def predict_transmission(self, x: np.ndarray, offsets: dict | None = None) -> dict:
        return {pid: np.exp(-lam) for pid, lam in self.predict_opacity(x, offsets).items()}


def build_forward_model(
    geom: GeometryConfig,
    txedges: np.ndarray,
    tyedges: np.ndarray,
    grid: VoxelGrid | None = None,
    cache_dir: str | None = "runs/.cache",
) -> ForwardModel:
    txc = 0.5 * (txedges[:-1] + txedges[1:])
    tyc = 0.5 * (tyedges[:-1] + tyedges[1:])
    if grid is None:
        grid = auto_grid(geom, t_max=float(max(abs(txedges[0]), abs(txedges[-1]))))
    poses = {pid: geom.pose(pid) for pid in geom.poses}
    A = build_system_matrix(
        poses, txc, tyc, grid,
        aperture_m=geom.aperture_m, n_sub=geom.n_aperture_sub,
        detector_height_m=geom.detector_height_m, cache_dir=cache_dir,
    )
    return ForwardModel(A=A, grid=grid, pose_ids=list(poses), txedges=txedges, tyedges=tyedges)

"""World-frame geometry: detector poses, voxel grid, angular-bin ray bundles.

World frame: pos0 detector center at the origin, z pointing up, meters.
A track in angular bin (tx, ty) travels along direction (tx, ty, 1) in the
detector frame; a pose adds a position offset and a yaw rotation about z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig, PoseConfig


@dataclass(frozen=True)
class VoxelGrid:
    origin: tuple  # (x0, y0, z0) of the grid corner, meters
    spacing: float
    shape: tuple  # (nx, ny, nz)

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def axis_centers(self, axis: int) -> np.ndarray:
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * self.spacing

    def extent(self, axis: int) -> tuple[float, float]:
        return self.origin[axis], self.origin[axis] + self.shape[axis] * self.spacing

    def key(self) -> str:
        return f"{self.origin}-{self.spacing}-{self.shape}"


def pose_rotation(pose: PoseConfig) -> np.ndarray:
    """Rotation matrix mapping detector-frame directions to world frame (yaw about z)."""
    c, s = np.cos(np.radians(pose.yaw_deg)), np.sin(np.radians(pose.yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def bin_directions(txcenters: np.ndarray, tycenters: np.ndarray, pose: PoseConfig) -> np.ndarray:
    """Unit world-frame direction per angular bin -> array [ntx, nty, 3]."""
    tx, ty = np.meshgrid(txcenters, tycenters, indexing="ij")
    d = np.stack([tx, ty, np.ones_like(tx)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ pose_rotation(pose).T


def effective_aperture(t: np.ndarray, aperture_m: float, height_m: float) -> np.ndarray:
    """Width of the accepted ray-position window at angle tan(theta) = t.

    A track must cross both the top and bottom planes (separation height_m), so
    the accepted window at the detector mid-plane is centered and shrinks as
    a - |t| * H. Clamped to a small positive width.
    """
    return np.maximum(aperture_m - np.abs(t) * height_m, 0.02 * aperture_m)


def aperture_offsets(
    pose: PoseConfig,
    aperture_m: float,
    n_sub: int,
    tx: np.ndarray | None = None,
    ty: np.ndarray | None = None,
    height_m: float = 0.0,
) -> np.ndarray:
    """World-frame start offsets of the n_sub^2 parallel sub-rays of a ray bundle.

    With tx/ty (per-bin arrays) and height_m given, the window is the
    angle-dependent 4-layer coincidence overlap -> returns [nbin, n_sub^2, 3];
    otherwise the full square aperture -> [n_sub^2, 3].
    """
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    ux, uy = np.meshgrid(u, u, indexing="ij")
    ux, uy = ux.ravel(), uy.ravel()
    R = pose_rotation(pose).T
    if tx is None:
        offs = np.stack([ux * aperture_m, uy * aperture_m, np.zeros_like(ux)], axis=-1)
        return offs @ R
    wx = effective_aperture(np.asarray(tx), aperture_m, height_m)[:, None]
    wy = effective_aperture(np.asarray(ty), aperture_m, height_m)[:, None]
    offs = np.stack(
        [ux[None, :] * wx, uy[None, :] * wy, np.zeros((len(wx), len(ux)))], axis=-1
    )
    return offs @ R


def auto_grid(geom: GeometryConfig, t_max: float) -> VoxelGrid:
    """Voxel grid covering the union of all pose ray footprints over the z range."""
    z0, z1 = geom.grid_z_m
    if geom.grid_xy_m is not None:
        (x0, x1), (y0, y1) = geom.grid_xy_m
    else:
        half = geom.aperture_m / 2
        xs, ys = [], []
        for pid in geom.poses:
            p = geom.pose(pid)
            # footprint of |t| <= t_max rays at the top of the grid, ignoring yaw (bound)
            reach = t_max * z1 + half
            xs += [p.x - reach, p.x + reach]
            ys += [p.y - reach, p.y + reach]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sp = geom.grid_spacing_m
    nx = max(1, int(np.ceil((x1 - x0) / sp)))
    ny = max(1, int(np.ceil((y1 - y0) / sp)))
    nz = max(1, int(np.ceil((z1 - z0) / sp)))
    return VoxelGrid(origin=(float(x0), float(y0), float(z0)), spacing=sp, shape=(nx, ny, nz))

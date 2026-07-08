"""Sparse system matrix: mean path length of each angular bin's ray bundle per voxel.

The detector aperture (~65 cm) is comparable to the ceiling-beam pitch at ~3 m, so
each angular bin is modeled as a bundle of n_sub^2 parallel sub-rays spread across
the aperture; matrix entries are path lengths averaged over the bundle.

Rows cover ALL bins of the cropped angular window for every pose (masked-bin
selection is a row slice at solve time), so one cached matrix serves full fits,
holdout fits, and single-position cross-validation fits alike.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from .config import PoseConfig
from .geometry import VoxelGrid, aperture_offsets, bin_directions

_SAMPLES_PER_VOXEL = 3  # sampling step = spacing / this


def build_system_matrix(
    poses: dict[str, PoseConfig],
    txcenters: np.ndarray,
    tycenters: np.ndarray,
    grid: VoxelGrid,
    aperture_m: float = 0.65,
    n_sub: int = 4,
    detector_height_m: float = 0.8,
    cache_dir: str | Path | None = None,
) -> sparse.csr_matrix:
    """A[(pose, tx, ty) row-major, voxel] in meters. Pose order = dict order.

    detector_height_m sets the angle-dependent effective aperture (the 4-layer
    coincidence overlap, width aperture - |t| * height); pass 0 for the full
    square aperture at every angle.
    """
    if cache_dir is not None:
        key = _cache_key(poses, txcenters, tycenters, grid, aperture_m, n_sub, detector_height_m)
        cache = Path(cache_dir) / f"A_{key}.npz"
        if cache.exists():
            return sparse.load_npz(cache)
    blocks = [
        _pose_block(pose, txcenters, tycenters, grid, aperture_m, n_sub, detector_height_m)
        for pose in poses.values()
    ]
    A = sparse.vstack(blocks, format="csr")
    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, A)
    return A


def _pose_block(
    pose: PoseConfig,
    txcenters: np.ndarray,
    tycenters: np.ndarray,
    grid: VoxelGrid,
    aperture_m: float,
    n_sub: int,
    detector_height_m: float,
) -> sparse.csr_matrix:
    dirs = bin_directions(txcenters, tycenters, pose).reshape(-1, 3)  # [nb, 3]
    tx, ty = np.meshgrid(txcenters, tycenters, indexing="ij")
    offs = aperture_offsets(
        pose, aperture_m, n_sub, tx=tx.ravel(), ty=ty.ravel(), height_m=detector_height_m
    )  # [nb, ns, 3]
    nb, ns = len(dirs), offs.shape[1]
    origin = np.asarray(grid.origin)
    z0, z1 = grid.extent(2)

    # Path length of each bin's central axis inside the z slab of the grid.
    dz = dirs[:, 2]
    L = (z1 - z0) / dz
    t_in = (z0 - pose.z) / dz
    step = grid.spacing / _SAMPLES_PER_VOXEL
    n_samp = np.maximum(np.ceil(L / step).astype(np.int64), 1)

    rows, cols, vals = [], [], []
    start0 = np.array([pose.x, pose.y, pose.z]) + offs  # [nb, ns, 3]
    for lo in range(0, nb, 512):
        hi = min(lo + 512, nb)
        cn = n_samp[lo:hi]
        bin_rep = np.repeat(np.arange(lo, hi), cn)  # [Nt]
        # midpoint sampling fractions along each ray's in-grid segment
        frac = (np.arange(cn.sum()) - np.repeat(np.concatenate([[0], np.cumsum(cn[:-1])]), cn) + 0.5) / np.repeat(cn, cn)
        t = t_in[bin_rep] + frac * L[bin_rep]  # [Nt]
        pts = start0[bin_rep] + (t[:, None] * dirs[bin_rep])[:, None, :]  # [Nt, ns, 3]
        idx = np.floor((pts - origin) / grid.spacing).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=-1)  # [Nt, ns]
        flat = (idx[..., 0] * grid.shape[1] + idx[..., 1]) * grid.shape[2] + idx[..., 2]
        dl = (L[bin_rep] / n_samp[bin_rep] / ns).astype(np.float64)  # [Nt]
        r = np.broadcast_to(bin_rep[:, None], ok.shape)[ok]
        rows.append(r)
        cols.append(flat[ok])
        vals.append(np.broadcast_to(dl[:, None], ok.shape)[ok])
    A = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nb, grid.n_voxels),
    )
    return A.tocsr()


def _cache_key(poses, txc, tyc, grid, aperture_m, n_sub, detector_height_m) -> str:
    h = hashlib.sha256()
    for pid, p in poses.items():
        h.update(f"{pid}:{p.x:.4f},{p.y:.4f},{p.z:.4f},{p.yaw_deg:.3f};".encode())
    h.update(np.ascontiguousarray(txc).tobytes())
    h.update(np.ascontiguousarray(tyc).tobytes())
    h.update(f"{grid.key()}|{aperture_m}|{n_sub}|{detector_height_m}".encode())
    return h.hexdigest()[:16]

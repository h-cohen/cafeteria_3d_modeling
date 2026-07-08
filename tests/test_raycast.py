"""System-matrix correctness: path lengths, adjoint identity, phantom round trip."""

import numpy as np
from scipy import sparse

from muontomo.config import GeometryConfig, PoseConfig
from muontomo.forward import build_forward_model
from muontomo.geometry import VoxelGrid
from muontomo.raycast import build_system_matrix


def _pencil_model(pose=None, spacing=0.1):
    """Single-pose model with a pointlike aperture (pencil rays) for exact checks."""
    geom = GeometryConfig(
        poses={"pos0": pose or PoseConfig()},
        grid_z_m=(1.0, 3.0),
        grid_xy_m=((-2.0, 2.0), (-2.0, 2.0)),
        grid_spacing_m=spacing,
        aperture_m=1e-6,
        n_aperture_sub=1,
    )
    edges = np.linspace(-0.5, 0.5, 11)
    return build_forward_model(geom, edges, edges, cache_dir=None)


def test_vertical_ray_path_length():
    fwd = _pencil_model()
    # central bin: tx = ty = 0.05 (bin center); path length through slab = depth/cos ~ known
    row = fwd.A[fwd.rows("pos0")].toarray().reshape(10, 10, -1)
    for i, j in [(5, 5), (0, 0), (9, 3)]:
        tx = 0.5 * (fwd.txedges[i] + fwd.txedges[i + 1])
        ty = 0.5 * (fwd.tyedges[j] + fwd.tyedges[j + 1])
        expected = 2.0 * np.sqrt(1 + tx**2 + ty**2)  # slab 1..3 m
        assert np.isclose(row[i, j].sum(), expected, rtol=1e-3)


def test_uniform_volume_opacity_scales_with_secant():
    fwd = _pencil_model()
    x = np.ones(fwd.grid.n_voxels)
    lam = fwd.predict_opacity(x)["pos0"]
    tx, ty = np.meshgrid(
        0.5 * (fwd.txedges[:-1] + fwd.txedges[1:]),
        0.5 * (fwd.tyedges[:-1] + fwd.tyedges[1:]),
        indexing="ij",
    )
    assert np.allclose(lam, 2.0 * np.sqrt(1 + tx**2 + ty**2), rtol=1e-3)


def test_adjoint_identity():
    fwd = _pencil_model(spacing=0.25)
    rng = np.random.default_rng(0)
    x = rng.normal(size=fwd.A.shape[1])
    y = rng.normal(size=fwd.A.shape[0])
    assert np.isclose((fwd.A @ x) @ y, x @ (fwd.A.T @ y), rtol=1e-10)


def test_pose_offset_shifts_footprint():
    a = build_system_matrix(
        {"p": PoseConfig()}, np.array([0.0]), np.array([0.0]),
        VoxelGrid(origin=(-2, -2, 1), spacing=0.5, shape=(8, 8, 2)),
        aperture_m=1e-6, n_sub=1,
    ).toarray().reshape(8, 8, 2)
    b = build_system_matrix(
        {"p": PoseConfig(x=1.0)}, np.array([0.0]), np.array([0.0]),
        VoxelGrid(origin=(-2, -2, 1), spacing=0.5, shape=(8, 8, 2)),
        aperture_m=1e-6, n_sub=1,
    ).toarray().reshape(8, 8, 2)
    # a vertical pencil ray hits one xy column; moving the pose by 1 m = 2 voxels
    assert np.allclose(np.roll(a, 2, axis=0), b)


def test_aperture_spreads_ray_bundle():
    grid = VoxelGrid(origin=(-1, -1, 1), spacing=0.1, shape=(20, 20, 1))
    A = build_system_matrix(
        {"p": PoseConfig()}, np.array([0.0]), np.array([0.0]), grid,
        aperture_m=0.65, n_sub=4,
    )
    hit = (A.toarray().reshape(20, 20)[:, :] > 0).sum()
    # 16 sub-rays across 0.65 m -> roughly 6-7 x-columns (x n_sub y-columns) touched
    assert hit >= 16
    assert np.isclose(A.sum(), 0.1, rtol=1e-6)  # total path conserved: 1 voxel depth


def test_matrix_cache_roundtrip(tmp_path):
    grid = VoxelGrid(origin=(-1, -1, 1), spacing=0.5, shape=(4, 4, 2))
    args = ({"p": PoseConfig()}, np.array([0.0, 0.2]), np.array([0.0]), grid)
    a = build_system_matrix(*args, cache_dir=tmp_path)
    b = build_system_matrix(*args, cache_dir=tmp_path)
    assert (a != b).nnz == 0
    assert len(list(tmp_path.glob("A_*.npz"))) == 1

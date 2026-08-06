"""The single data-loading + calibration front-end shared by every enhancer.

load_context(run_dir) does ALL the I/O and sky calibration once, exposing:
  - the calibrated data (tmaps, omaps)          -- the calibration step, one place
  - the reconstruction ceiling-layer slice      -- what techniques sharpen
  - the model-free measured backprojection guide -- the "2D flux" reference
  - a thin-layer forward model + its measurements -- for data-consistent methods

No technique re-loads or re-calibrates anything; they only read an EnhanceContext.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ..backproject import backproject_opacity
from ..calibration import transmission_maps
from ..config import RunConfig
from ..io import load_dataset
from ..opacity import opacity_map


@dataclass
class LayerModel:
    """Thin single-layer forward model + its aligned measurements (all on the
    main volume's xy grid, since _layer_model inherits grid_xy_m)."""

    fwd: object  # ForwardModel over grid (nx, ny, nz_thin)
    lam: np.ndarray  # [n_rows] measured opacity
    w: np.ndarray  # [n_rows] least-squares weights (0 on masked/holdout bins)
    pose_of_row: np.ndarray  # [n_rows] pose index per row
    nz_thin: int  # number of z-voxels spanning the layer thickness


@dataclass
class EnhanceContext:
    run: Path
    cfg: RunConfig
    tmaps: dict
    omaps: dict
    rho: np.ndarray
    origin: tuple
    spacing: float
    z_layer: float
    iz: int
    xs: np.ndarray
    ys: np.ndarray
    layer: np.ndarray  # reconstruction slice at z_layer, main grid (nx, ny), >= 0
    guide: np.ndarray  # mean measured backprojection on (xs, ys) -- the combined data view
    per_pose_guide: dict  # {pose_id: single-detector backprojection}, same grid
    sharp_guide_id: str  # pose whose backprojection has the crispest beams
    _cache_dir: str | None = field(default="runs/.cache", repr=False)

    @property
    def sharp_guide(self) -> np.ndarray:
        """The single cleanest detector's backprojection -- used as the edge
        reference for guided filtering. The MEAN dilutes a beam that only the
        well-covering detector sees; the sharpest single view keeps all beams."""
        return self.per_pose_guide[self.sharp_guide_id]

    # ---- lazily built pieces (only the data-consistent methods need them) ----
    @cached_property
    def layer_model(self) -> LayerModel:
        from ..reconstruct import _layer_model, fit_data

        rc = self.cfg.reconstruction
        tm0 = next(iter(self.tmaps.values()))
        lf = _layer_model(
            self.cfg.geometry, tm0.txedges, tm0.tyedges,
            self.z_layer, rc.layered_thickness, cache_dir=self._cache_dir,
        )
        data = fit_data(lf, self.omaps)
        return LayerModel(
            fwd=lf, lam=data.lam, w=data.w, pose_of_row=data.pose_of_row,
            nz_thin=lf.grid.shape[2],
        )

    def collapse_layer(self, x_thin: np.ndarray) -> np.ndarray:
        """A thin-layer volume (nx, ny, nz_thin) -> the 2D layer used everywhere
        else. Mean over z keeps the fitted opacity density (each z-voxel holds
        the same density in the pipeline's embedding)."""
        return np.asarray(x_thin).reshape(self.layer_model.fwd.grid.shape).mean(axis=2)

    def display_blur(self, img: np.ndarray, sigma_m: float = 0.12) -> np.ndarray:
        """The viewer's light in-plane smoothing, so enhancer inputs/outputs are
        compared on the same footing as what the viewer shows."""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(np.asarray(img), sigma_m / self.spacing)


def load_context(run_dir: str | Path, cache_dir: str | None = "runs/.cache") -> EnhanceContext:
    run = Path(run_dir)
    cfg = RunConfig.load(run / "config.json")
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    omaps = {pid: opacity_map(t) for pid, t in tmaps.items()}

    with np.load(run / "volume.npz") as vz:
        rho = np.maximum(vz["rho"].astype(np.float64), 0.0)
        origin = tuple(float(v) for v in vz["origin"])
        spacing = float(vz["spacing"])

    if cfg.reconstruction.layered_zs:
        z_layer = float(cfg.reconstruction.layered_zs[0])
        iz = int(round((z_layer - origin[2]) / spacing))
    else:
        iz = int(np.argmax(rho.sum(axis=(0, 1))))
        z_layer = origin[2] + iz * spacing
    iz = max(0, min(iz, rho.shape[2] - 1))

    nx, ny = rho.shape[0], rho.shape[1]
    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing
    layer = rho[:, :, iz]
    per_pose, guide = backproject_opacity(tmaps, cfg.geometry, z_layer, xs, ys)
    per_pose = {pid: np.nan_to_num(v, nan=0.0) for pid, v in per_pose.items()}
    guide = np.nan_to_num(guide, nan=0.0)

    # sharpest detector = highest central-band stripe contrast (most beam-like)
    band = (ys > -2.0) & (ys < 2.0)
    core = (xs > cfg.geometry.grid_xy_m[0][0] + 1.0) & (xs < cfg.geometry.grid_xy_m[0][1] - 1.0)

    def _contrast(g):
        p = g[:, band].mean(axis=1)[core]
        return (np.percentile(p, 90) - np.percentile(p, 10)) / (np.percentile(p, 90) + 1e-9)

    sharp_id = max(per_pose, key=lambda pid: _contrast(per_pose[pid]))

    return EnhanceContext(
        run=run, cfg=cfg, tmaps=tmaps, omaps=omaps, rho=rho, origin=origin,
        spacing=spacing, z_layer=z_layer, iz=iz, xs=xs, ys=ys,
        layer=layer, guide=guide, per_pose_guide=per_pose, sharp_guide_id=sharp_id,
        _cache_dir=cache_dir,
    )

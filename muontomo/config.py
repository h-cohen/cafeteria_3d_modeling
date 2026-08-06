"""Run configuration: every knob of the pipeline in one JSON-serializable object.

Conventions frozen from Stage-0 inspection of the ROOT files:
  * `txty` is 800x800 over tan(theta_x), tan(theta_y) in [-2, 2]; axis0 = tx, axis1 = ty.
  * Track counts: sky 28.5M, pos0 6.3M, pos1 2.4M.
  * World frame: pos0 detector center at origin, z up (toward the ceiling), meters.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PoseConfig:
    """Detector pose: position of the detector center and yaw about z (degrees)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class GeometryConfig:
    # User-reported prior: the detector moved ~1.5-2 m horizontally between positions
    # (same floor, so identical z). Refined by muontomo.selfcal.
    poses: dict = field(default_factory=lambda: {"pos0": PoseConfig(), "pos1": PoseConfig(x=1.75)})
    ceiling_z_prior_m: float = 3.0
    # Voxel grid: z range fixed; xy extent derived from ray footprints unless given.
    grid_z_m: tuple = (1.5, 5.0)
    grid_spacing_m: float = 0.10
    grid_xy_m: tuple | None = None  # ((x0,x1),(y0,y1)) or None -> auto from footprints
    # Sub-box of grid_xy_m to actually display in the 3D viewer, or None -> show
    # all of grid_xy_m. The solver grid is deliberately kept wide so the
    # limited-angle edge-of-coverage bias (SIRT/TV pools spurious mass at
    # whichever boundary is worst-constrained) lands away from the real
    # structure; this lets the viewer crop that margin out without shrinking
    # the grid the reconstruction actually runs on.
    viewer_crop_xy_m: tuple | None = None
    aperture_m: float = 0.65
    n_aperture_sub: int = 4  # sub-rays per axis across the aperture (n^2 total)
    detector_height_m: float = 0.8  # layer separation; sets the angle-dependent aperture

    def pose(self, pid: str) -> PoseConfig:
        p = self.poses[pid]
        return p if isinstance(p, PoseConfig) else PoseConfig(**p)


@dataclass
class BinningConfig:
    hist: str = "txty"
    t_max: float = 1.0  # crop |tx|,|ty| to this before rebinning
    rebin: int = 8  # 800 bins over +-2 -> crop to +-1 (400) -> 50x50 of width 0.04


@dataclass
class CalibrationConfig:
    min_sky: float = 25.0
    min_cafe: float = 5.0
    # Transmission scale: quantile of smoothed T mapped to 1 ("near-open sky" bins).
    norm_quantile: float = 0.95
    norm_smooth_bins: float = 1.0


@dataclass
class ReconstructionConfig:
    algorithm: str = "tv"  # sirt | tv | mlem | layered
    n_iter: int = 150
    nonneg: bool = True
    chi2_target: float = 1.0  # discrepancy-principle stop for sirt/mlem
    tv_alpha: float = 0.01  # SIRT-TV denoise threshold, as a fraction of x's p95
    tv_z_weight: float = 0.5  # anisotropic TV: relative weight of z gradients
    # 'layered': candidate layer heights (m) scanned and scored by cross-position CV
    layered_zs: tuple = (2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4)
    layered_thickness: float = 0.3  # slab thickness of each fitted layer (m)
    holdout_fraction: float = 0.2  # random-bin holdout for validation metrics
    seed: int = 42


@dataclass
class RunConfig:
    data: dict = field(
        default_factory=lambda: {
            "sky": "data/HistsOutSkyRoofRuns37-77.root",
            "pos0": "data/HistsOutDataCafePos0.root",
            "pos1": "data/HistsOutDataCafePos1.root",
        }
    )
    binning: BinningConfig = field(default_factory=BinningConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    notes: str = ""

    # ---- serialization ----

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        d.pop("_snapshot", None)
        kwargs: dict = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name == "binning":
                v = BinningConfig(**v)
            elif f.name == "calibration":
                v = CalibrationConfig(**v)
            elif f.name == "geometry":
                v = GeometryConfig(**_tupled(v, ("grid_z_m", "grid_xy_m", "viewer_crop_xy_m")))
            elif f.name == "reconstruction":
                v = ReconstructionConfig(**_tupled(v, ("layered_zs",)))
            kwargs[f.name] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path, snapshot: bool = True) -> None:
        d = self.to_dict()
        if snapshot:
            d["_snapshot"] = {"config_hash": self.hash(), "git": _git_hash()}
        Path(path).write_text(json.dumps(d, indent=2, default=_json_default) + "\n")

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _tupled(d: dict, keys: tuple) -> dict:
    d = dict(d)
    for k in keys:
        if d.get(k) is not None:
            v = d[k]
            d[k] = tuple(tuple(e) if isinstance(e, list) else e for e in v)
    return d


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"

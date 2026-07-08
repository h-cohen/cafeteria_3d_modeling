"""Synthetic phantoms: known room volumes forward-projected into fake count histograms.

The phantom is the only rigorous ground truth in this project. It exercises the
exact same forward model and file API as the real data, including imprecisely
known detector poses: reconstruction is handed a deliberately wrong pose
(`pose_error`) while the counts are generated with the true one.

Usage:
    python -m muontomo.phantom --preset beams --seed 42 --out phantoms/p1
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .config import GeometryConfig, PoseConfig, RunConfig
from .forward import build_forward_model
from .geometry import VoxelGrid

PRESETS: dict[str, dict] = {
    "slab": {
        "slab": {"z0": 2.9, "thickness": 0.25, "kappa": 0.8},
        "beams": None,
    },
    "beams": {},
    "beams_box": {
        "objects": [{"type": "box", "center": [1.0, 0.5, 1.2], "size": [0.6, 0.6, 1.0], "kappa": 0.5}],
    },
    "beams_lowstat": {"counts": {"sky": 2_500_000, "pos0": 600_000, "pos1": 240_000}},
}

DEFAULT_SPEC: dict = {
    "grid": {"spacing_m": 0.10, "z_m": [1.5, 5.0]},
    # Finite room: ceiling structure ends at the walls. Real rooms are finite, and the
    # edges are what breaks the parallax aliasing of a periodic beam pattern.
    "room": {"x": [-3.0, 4.5], "y": [-3.0, 3.5]},
    "slab": {"z0": 3.1, "thickness": 0.15, "kappa": 0.6},
    "beams": {"pitch": 0.60, "width": 0.20, "depth": 0.30, "direction_deg": 0.0, "phase": 0.1, "kappa": 0.9},
    "objects": [],
    "geometry": {
        "poses": {"pos0": {"x": 0.0, "y": 0.0}, "pos1": {"x": 2.0, "y": 0.0}},
        "pose_error": {"pos1": {"x": 0.3, "yaw_deg": 3.0}},
    },
    "binning": {"t_max": 1.0, "n_bins": 50},
    # Match the real campaign: sky 28.5M, pos0 6.3M, pos1 2.4M tracks.
    "counts": {"sky": 28_500_000, "pos0": 6_300_000, "pos1": 2_400_000},
    "sky_cos_power": 2.0,
    "seed": 42,
}


def make_spec(preset: str) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    spec = copy.deepcopy(DEFAULT_SPEC)
    for k, v in PRESETS[preset].items():
        spec[k] = copy.deepcopy(v)
    spec["preset"] = preset
    return spec


def rasterize_volume(spec: dict, grid: VoxelGrid) -> np.ndarray:
    """Truth volume of opacity density kappa [1/m] on the grid."""
    x = grid.axis_centers(0)[:, None, None]
    y = grid.axis_centers(1)[None, :, None]
    z = grid.axis_centers(2)[None, None, :]
    vol = np.zeros(grid.shape, dtype=np.float64)

    room = spec.get("room")
    if room:
        in_room = (x >= room["x"][0]) & (x <= room["x"][1]) & (y >= room["y"][0]) & (y <= room["y"][1])
    else:
        in_room = np.ones((), dtype=bool)

    slab = spec.get("slab")
    if slab:
        in_slab = (z >= slab["z0"]) & (z < slab["z0"] + slab["thickness"]) & in_room
        vol += slab["kappa"] * np.broadcast_to(in_slab, vol.shape).astype(float)

    beams = spec.get("beams")
    if beams:
        ang = np.radians(beams["direction_deg"])
        u = x * np.cos(ang) + y * np.sin(ang)  # coordinate across the beam direction
        u_mod = np.mod(u - beams["phase"], beams["pitch"])
        under = u_mod < beams["width"]
        z0 = (slab["z0"] if slab else 3.1) - beams["depth"]
        in_beam = under & (z >= z0) & (z < z0 + beams["depth"]) & in_room
        vol += beams["kappa"] * np.broadcast_to(in_beam, vol.shape).astype(float)

    for obj in spec.get("objects") or []:
        c, s = np.asarray(obj["center"]), np.asarray(obj["size"])
        if obj["type"] == "box":
            inside = (
                (np.abs(x - c[0]) < s[0] / 2)
                & (np.abs(y - c[1]) < s[1] / 2)
                & (np.abs(z - c[2]) < s[2] / 2)
            )
        elif obj["type"] == "sphere":
            inside = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) < (s[0] / 2) ** 2
        else:
            raise ValueError(f"unknown object type {obj['type']!r}")
        vol += obj["kappa"] * np.broadcast_to(inside, vol.shape).astype(float)
    return vol


def _sky_template(txc: np.ndarray, tyc: np.ndarray, cos_power: float, total: float) -> np.ndarray:
    """Expected open-sky counts per angular bin: cos(theta)^n acceptance-like shape."""
    tx, ty = np.meshgrid(txc, tyc, indexing="ij")
    cos_theta = 1.0 / np.sqrt(1.0 + tx**2 + ty**2)
    shape = cos_theta ** (cos_power + 3)  # flux cos^n * solid-angle Jacobian cos^3
    return shape / shape.sum() * total


def true_geometry(spec: dict) -> GeometryConfig:
    poses = {pid: PoseConfig(**p) for pid, p in spec["geometry"]["poses"].items()}
    g = spec["geometry"]
    z_m = tuple(spec["grid"]["z_m"])
    return GeometryConfig(poses=poses, grid_z_m=z_m, grid_spacing_m=spec["grid"]["spacing_m"])


def reported_geometry(spec: dict) -> GeometryConfig:
    """The (wrong) geometry handed to reconstruction: true pose + pose_error."""
    geom = true_geometry(spec)
    for pid, err in spec["geometry"].get("pose_error", {}).items():
        p = geom.pose(pid)
        geom.poses[pid] = PoseConfig(
            x=p.x + err.get("x", 0.0),
            y=p.y + err.get("y", 0.0),
            z=p.z + err.get("z", 0.0),
            yaw_deg=p.yaw_deg + err.get("yaw_deg", 0.0),
        )
    return geom


def generate(spec: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["seed"])

    nb, t_max = spec["binning"]["n_bins"], spec["binning"]["t_max"]
    edges = np.linspace(-t_max, t_max, nb + 1)
    txc = 0.5 * (edges[:-1] + edges[1:])

    geom = true_geometry(spec)
    fwd = build_forward_model(geom, edges, edges, cache_dir=None)
    truth = rasterize_volume(spec, fwd.grid)
    trans = fwd.predict_transmission(truth)

    sky_mu = _sky_template(txc, txc, spec["sky_cos_power"], spec["counts"]["sky"])
    n_sky = rng.poisson(sky_mu).astype(np.float64)
    _save_counts(out / "counts_sky.npz", n_sky, edges)
    for pid in fwd.pose_ids:
        mu = sky_mu * (spec["counts"][pid] / spec["counts"]["sky"]) * trans[pid]
        _save_counts(out / f"counts_{pid}.npz", rng.poisson(mu).astype(np.float64), edges)

    np.savez_compressed(
        out / "truth_volume.npz",
        rho=truth.astype(np.float32),
        origin=np.asarray(fwd.grid.origin),
        spacing=fwd.grid.spacing,
        shape=np.asarray(fwd.grid.shape),
    )
    meta = {
        "spec": spec,
        "true_poses": {pid: vars(geom.pose(pid)) for pid in geom.poses},
        "reported_poses": {pid: vars(reported_geometry(spec).pose(pid)) for pid in geom.poses},
        "grid": {"origin": list(fwd.grid.origin), "spacing": fwd.grid.spacing, "shape": list(fwd.grid.shape)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def _save_counts(path: Path, values: np.ndarray, edges: np.ndarray) -> None:
    np.savez_compressed(path, values=values, xedges=edges, yedges=edges, name="txty")


def phantom_run_config(phantom_dir: str | Path) -> RunConfig:
    """RunConfig for reconstructing a phantom with the REPORTED (wrong) poses."""
    meta = json.loads((Path(phantom_dir) / "meta.json").read_text())
    spec = meta["spec"]
    cfg = RunConfig(data={"phantom": str(phantom_dir)})
    cfg.binning.hist = "txty"
    cfg.binning.t_max = spec["binning"]["t_max"]
    cfg.binning.rebin = 1  # phantom counts are already at analysis binning
    cfg.geometry = reported_geometry(spec)
    cfg.geometry.ceiling_z_prior_m = spec["slab"]["z0"] if spec.get("slab") else 3.0
    # pin the reconstruction xy grid to the truth grid so volumes align exactly
    g = meta["grid"]
    x0, y0 = g["origin"][0], g["origin"][1]
    nx, ny = g["shape"][0], g["shape"][1]
    sp = g["spacing"]
    cfg.geometry.grid_xy_m = ((x0, x0 + nx * sp), (y0, y0 + ny * sp))
    cfg.geometry.grid_spacing_m = sp
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="beams", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = make_spec(args.preset)
    if args.seed is not None:
        spec["seed"] = args.seed
    out = generate(spec, args.out)
    print(f"phantom '{args.preset}' written to {out}")


if __name__ == "__main__":
    main()

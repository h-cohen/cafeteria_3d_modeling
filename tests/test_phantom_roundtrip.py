"""End-to-end: phantom -> counts -> calibration -> reconstruction recovers the truth."""

import copy
import json

import numpy as np
import pytest

from muontomo import phantom
from muontomo.calibration import transmission_maps
from muontomo.config import RunConfig
from muontomo.io import load_dataset
from muontomo.reconstruct import produce_run


@pytest.fixture(scope="module")
def small_phantom(tmp_path_factory):
    """Coarse, exactly-posed phantom for fast round-trip checks."""
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.2, "z_m": [1.5, 4.5]}
    spec["binning"] = {"t_max": 0.8, "n_bins": 25}
    spec["geometry"]["pose_error"] = {}
    out = tmp_path_factory.mktemp("phantoms") / "p_small"
    phantom.generate(spec, out)
    return out, spec


def test_phantom_counts_encode_truth_transmission(small_phantom):
    """Measured/sky count ratio must match exp(-lambda_truth) bin by bin (up to noise)."""
    out, spec = small_phantom
    ds = load_dataset({"phantom": str(out)})
    cfg = phantom.phantom_run_config(out)
    tmaps = transmission_maps(ds, cfg.binning, cfg.calibration)

    with np.load(out / "truth_volume.npz") as z:
        truth = z["rho"].astype(np.float64)
    from muontomo.forward import build_forward_model

    fwd = build_forward_model(phantom.true_geometry(spec),
                              tmaps["pos0"].txedges, tmaps["pos0"].tyedges, cache_dir=None)
    t_true = fwd.predict_transmission(truth.ravel())["pos0"]
    tm = tmaps["pos0"]
    good = tm.mask & (tm.n_sky > 2000)
    # The estimated scale is biased when no bin sees open sky (the c_pose nuisance
    # absorbs this downstream); compare against the exact generation scale here.
    s_true = spec["counts"]["pos0"] / spec["counts"]["sky"]
    T_meas = tm.T * (tm.scale / s_true)
    sigma = tm.sigma_T * (tm.scale / s_true)
    # pull distribution: (T_meas - T_true)/sigma should be ~N(0,1)
    pulls = (T_meas[good] - t_true[good]) / sigma[good]
    assert abs(np.mean(pulls)) < 0.2
    assert np.std(pulls) < 1.5


def _ceiling_slices(run_dir, truth_path, z_true):
    """Aligned (recon, truth) ceiling-band xy slices; grids share xy layout."""
    with np.load(run_dir / "volume.npz") as z:
        rho = z["rho"].astype(np.float64)
        r_origin, spacing = z["origin"], float(z["spacing"])
    with np.load(truth_path) as z:
        truth = z["rho"].astype(np.float64)
        t_origin = z["origin"]
    zc_r = r_origin[2] + (np.arange(rho.shape[2]) + 0.5) * spacing
    zc_t = t_origin[2] + (np.arange(truth.shape[2]) + 0.5) * spacing
    band_r = (zc_r > z_true - 0.45) & (zc_r < z_true + 0.45)
    band_t = (zc_t > z_true - 0.45) & (zc_t < z_true + 0.45)
    rec = np.maximum(rho, 0)[:, :, band_r].sum(axis=2)
    tru = truth[:, :, band_t].sum(axis=2)
    assert rec.shape == tru.shape, "phantom_run_config must pin the xy grid"
    return rec, tru


# The slab's angular signature is height-independent (sec-theta at any height), so
# only the structured part localizes in z with 2 views; assertions therefore check
# the structure (slice correlation) for 3D solvers and the height for 'layered',
# which scans layer height scored by cross-position validation.
@pytest.mark.parametrize("algorithm,min_corr", [("sirt", 0.3), ("tv", 0.35), ("mlem", 0.2)])
def test_reconstruction_recovers_ceiling_structure(small_phantom, tmp_path, algorithm, min_corr):
    out, spec = small_phantom
    cfg = phantom.phantom_run_config(out)
    cfg.geometry.grid_z_m = (2.4, 3.8)  # ceiling prior +- 0.7 (excludes parallax aliases)
    cfg.reconstruction.algorithm = algorithm
    cfg.reconstruction.n_iter = 120
    cfg.reconstruction.tv_alpha = 0.01
    run = produce_run(cfg, tmp_path / f"run_{algorithm}", cache_dir=str(tmp_path / "cache"))

    z_true = spec["slab"]["z0"] - 0.15  # beams band center-of-mass
    rec, tru = _ceiling_slices(run, out / "truth_volume.npz", z_true)
    seen = rec + tru > 0
    r = np.corrcoef(rec[seen], tru[seen])[0, 1]
    assert r > min_corr, f"{algorithm}: ceiling-slice correlation {r:.3f}"


def test_layered_scan_finds_ceiling_height(small_phantom, tmp_path):
    out, spec = small_phantom
    cfg = phantom.phantom_run_config(out)
    cfg.geometry.grid_z_m = (2.4, 3.8)
    cfg.reconstruction.algorithm = "layered"
    cfg.reconstruction.n_iter = 80
    cfg.reconstruction.tv_alpha = 0.01
    cfg.reconstruction.layered_zs = (2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6)
    run = produce_run(cfg, tmp_path / "run_layered", cache_dir=str(tmp_path / "cache"))

    import json

    info = json.loads((run / "fit_info.json").read_text())
    z_hat = info["fits"]["full"]["layer_z"]
    # truth: beams 2.8-3.1, slab 3.1-3.25
    assert 2.7 <= z_hat <= 3.3, f"layered chose z={z_hat}"
    scan = info["fits"]["full"]["layer_scan"]
    assert len(scan) == 7 and all("cv_mean" in e for e in scan)


def test_reported_pose_carries_injected_error(tmp_path):
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.4, "z_m": [2.0, 4.0]}
    spec["binning"] = {"t_max": 0.6, "n_bins": 10}
    spec["counts"] = {"sky": 100_000, "pos0": 30_000, "pos1": 30_000}
    out = phantom.generate(spec, tmp_path / "p")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["reported_poses"]["pos1"]["x"] == meta["true_poses"]["pos1"]["x"] + 0.3
    assert meta["reported_poses"]["pos1"]["yaw_deg"] == 3.0
    cfg = phantom.phantom_run_config(out)
    assert cfg.geometry.pose("pos1").x == meta["reported_poses"]["pos1"]["x"]

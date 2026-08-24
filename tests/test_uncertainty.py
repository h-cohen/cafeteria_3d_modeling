"""Bootstrap uncertainty quantification (muontomo.uncertainty) on a tiny phantom."""

import numpy as np
import pytest

from muontomo import phantom
from muontomo.calibration import transmission_maps
from muontomo.io import load_dataset
from muontomo.reconstruct import produce_run
from muontomo.uncertainty import bootstrap, resample_tmaps


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.25, "z_m": [2.0, 4.0]}
    spec["binning"] = {"t_max": 0.8, "n_bins": 16}
    spec["geometry"]["pose_error"] = {}
    tmp = tmp_path_factory.mktemp("uq")
    out = phantom.generate(spec, tmp / "phantom")
    cfg = phantom.phantom_run_config(out)
    cfg.reconstruction.algorithm = "layered"
    cfg.reconstruction.layered_zs = (3.0,)
    cfg.reconstruction.n_iter = 30
    run = produce_run(cfg, tmp / "run", cache_dir=str(tmp / "cache"))
    return run, cfg, str(tmp / "cache")


def test_resample_preserves_geometry_and_perturbs_counts(small_run):
    run, cfg, _ = small_run
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    r = resample_tmaps(tmaps, np.random.default_rng(0))
    for pid, tm in tmaps.items():
        assert np.array_equal(r[pid].mask, tm.mask)  # identical bin geometry
        assert r[pid].scale == tm.scale
        assert not np.array_equal(r[pid].n_cafe, tm.n_cafe)  # counts perturbed
        # replica transmission stays close to the measurement on usable bins
        d = np.abs(r[pid].T - tm.T)[tm.mask]
        assert np.median(d) < 0.2


def test_bootstrap_outputs(small_run):
    run, _, cache = small_run
    r = bootstrap(run, n_layer=2, n_focus=2, seed=1, cache_dir=cache)
    assert (run / "uncertainty.json").exists()
    assert (run / "images" / "uncertainty.png").exists()
    assert r["n_layer_replicas"] == 2
    assert all(np.isfinite(b["pos_sigma_m"]) and b["pos_sigma_m"] >= 0 for b in r["beams"])
    af = r["autofocus"]
    assert 2.0 <= af["z_star_mean_m"] <= 4.0  # inside the scanned band
    assert af["z_star_sigma_m"] >= 0

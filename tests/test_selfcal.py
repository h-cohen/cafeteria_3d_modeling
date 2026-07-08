"""Stage B geometry self-calibration recovers an injected pose error on a phantom."""

import numpy as np
import pytest

from muontomo import phantom
from muontomo.selfcal import stage_b


@pytest.fixture(scope="module")
def misposed_phantom(tmp_path_factory):
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.25, "z_m": [2.0, 4.0]}
    spec["binning"] = {"t_max": 0.8, "n_bins": 16}
    spec["counts"] = {"sky": 3_000_000, "pos0": 800_000, "pos1": 500_000}
    spec["geometry"]["pose_error"] = {"pos1": {"x": 0.3, "yaw_deg": 3.0}}
    out = tmp_path_factory.mktemp("phantoms") / "misposed"
    phantom.generate(spec, out)
    return out, spec


def test_stage_b_recovers_offset_from_wrong_pose(misposed_phantom, tmp_path):
    out, spec = misposed_phantom
    cfg = phantom.phantom_run_config(out)  # geometry seeded with the WRONG (reported) pose
    true_x = spec["geometry"]["poses"]["pos1"]["x"]
    reported_x = cfg.geometry.pose("pos1").x
    assert not np.isclose(reported_x, true_x)  # sanity: the error really was injected

    est = stage_b(cfg, bounds_m=0.5, bounds_deg=6.0, cache_dir=str(tmp_path / "cache"))

    # Stage B starts from the wrong pose and must move back toward the true one.
    assert abs(est.dx - true_x) < abs(reported_x - true_x), (
        f"stage_b did not improve x: reported={reported_x:.3f} est={est.dx:.3f} true={true_x:.3f}"
    )
    assert abs(est.dx - true_x) < 0.2

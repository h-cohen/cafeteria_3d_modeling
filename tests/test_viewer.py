"""Build the viewer for a small phantom run and smoke-test it with Playwright."""

import pytest

from muontomo import phantom
from muontomo.reconstruct import produce_run
from muontomo.viewer.build import build_viewer
from muontomo.viewer.smoke import run_smoke


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.25, "z_m": [2.0, 4.0]}
    spec["binning"] = {"t_max": 0.8, "n_bins": 16}
    spec["geometry"]["pose_error"] = {}
    tmp = tmp_path_factory.mktemp("viewer")
    out = phantom.generate(spec, tmp / "phantom")
    cfg = phantom.phantom_run_config(out)
    cfg.reconstruction.algorithm = "sirt"
    cfg.reconstruction.n_iter = 60
    return produce_run(cfg, tmp / "run", cache_dir=str(tmp / "cache"))


def test_viewer_builds_and_renders(small_run):
    viewer = build_viewer(small_run)
    assert viewer.exists() and viewer.stat().st_size > 1000
    html = viewer.read_text()
    assert "THREE.WebGLRenderer" in html
    assert "__VOLUME__" in html


def test_viewer_smoke(small_run):
    build_viewer(small_run)
    report = run_smoke(small_run)
    assert not report["errors"]
    assert report["initial_state"]["triangles"] > 0
    assert len(report["screenshots"]) == 4

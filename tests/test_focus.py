"""Unit tests for the autofocus module (muontomo.focus).

All synthetic / mocked: no real data, no full SIRT solves (the one real
layer_cv_score call runs on a tiny 16-bin phantom).
"""

import numpy as np
import pytest

from muontomo import focus, phantom
from muontomo.calibration import transmission_maps
from muontomo.io import load_dataset


# ---------------------------------------------------------------- helpers

def _stripe_image(xs, ys, beam_xs, width=0.2, amp=1.0):
    """Synthetic ceiling: vertical stripes at beam_xs on a (nx, ny) grid."""
    img = np.zeros((xs.size, ys.size))
    for b in beam_xs:
        img += amp * np.exp(-0.5 * ((xs[:, None] - b) / width) ** 2)
    return img


@pytest.fixture(scope="module")
def tiny_phantom(tmp_path_factory):
    spec = phantom.make_spec("beams")
    spec["grid"] = {"spacing_m": 0.25, "z_m": [2.0, 4.0]}
    spec["binning"] = {"t_max": 0.8, "n_bins": 16}
    spec["geometry"]["pose_error"] = {}
    out = phantom.generate(spec, tmp_path_factory.mktemp("focus") / "phantom")
    cfg = phantom.phantom_run_config(out)
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    return cfg, tmaps


# ---------------------------------------------------------------- unit tests

def test_interior_mask_drops_margins():
    xs = np.linspace(-5, 5, 101)
    ys = np.linspace(-4, 4, 81)
    m = focus._interior_mask(xs, ys, margin_m=1.0)
    assert m.shape == (xs.size, ys.size)
    assert not m[0, 40] and not m[50, 0]  # edges masked
    assert m[50, 40]  # centre kept


def test_ncc_prefers_registered_images():
    rng = np.random.default_rng(0)
    xs = np.arange(-5, 5, 0.1) + 0.05
    ys = np.arange(-4, 4, 0.1) + 0.05
    img = _stripe_image(xs, ys, [-3.0, -1.5, 0.0, 1.5, 3.0])
    noisy = img + 0.05 * rng.standard_normal(img.shape)
    shifted = np.roll(noisy, 5, axis=0)  # 0.5 m lateral parallax
    mask = focus._interior_mask(xs, ys)
    assert focus._ncc(img, noisy, mask) > focus._ncc(img, shifted, mask)
    assert focus._ncc(img, img, mask) == pytest.approx(1.0, abs=1e-6)


def test_focus_curve_finds_registered_height():
    """Stacks where the two views register only at index k*: the NCC curve
    must peak there."""
    rng = np.random.default_rng(1)
    xs = np.arange(-5, 5, 0.1) + 0.05
    ys = np.arange(-4, 4, 0.1) + 0.05
    base = _stripe_image(xs, ys, [-3.0, -1.5, 0.0, 1.5, 3.0])
    zs = np.arange(6.0, 8.01, 0.25)
    k_true = 4  # zs[4] = 7.0
    a, b, m = [], [], []
    for k in range(zs.size):
        shift = 2 * (k - k_true)  # parallax grows away from the true height
        va = base + 0.05 * rng.standard_normal(base.shape)
        vb = np.roll(base, shift, axis=0) + 0.05 * rng.standard_normal(base.shape)
        a.append(va), b.append(vb), m.append(0.5 * (va + vb))
    curve = focus.focus_curve(np.stack(a), np.stack(b), np.stack(m), zs, xs, ys)
    assert curve["focus_z_m"] == pytest.approx(7.0)
    assert curve["focus_ncc"] > 0.9


def test_height_map_median_matches_global(tmp_path):
    rng = np.random.default_rng(2)
    xs = np.arange(-5, 5, 0.1) + 0.05
    ys = np.arange(-4, 4, 0.1) + 0.05
    base = _stripe_image(xs, ys, [-3.0, -1.5, 0.0, 1.5, 3.0])
    zs = np.arange(6.0, 8.01, 0.25)
    a, b = [], []
    for k in range(zs.size):
        shift = 2 * (k - 4)
        a.append(base + 0.05 * rng.standard_normal(base.shape))
        b.append(np.roll(base, shift, axis=0) + 0.05 * rng.standard_normal(base.shape))
    zmap, conf, stats = focus.height_map(np.stack(a), np.stack(b), zs, xs, ys)
    assert stats["median_z_m"] == pytest.approx(7.0, abs=0.3)
    assert 0 < stats["coverage_frac"] <= 1


def test_scan_zs_brackets_configured_height(tiny_phantom):
    cfg, _ = tiny_phantom
    cfg.reconstruction.layered_zs = (3.0,)
    zs = focus._scan_zs(cfg, step=0.2)
    assert zs.min() >= 1.0 and zs.max() == pytest.approx(4.0)
    assert 3.0 in np.round(zs, 2)


def test_cv_height_scan_two_stage_refines_minimum(tiny_phantom, monkeypatch):
    """With a mocked scorer (parabola with minimum at 3.03 m) the two-stage scan
    must return the fine-grid point nearest the minimum and a sorted scan that
    contains fine points around it."""
    cfg, tmaps = tiny_phantom
    cfg.reconstruction.layered_zs = (3.0,)

    calls = []

    def fake_score(geom, txe, tye, omaps, rc2, z, thickness, cache_dir=None,
                   cv_trim_pct=None, active=None):
        calls.append(float(z))
        return (float(z) - 3.03) ** 2 + 1.0, {}

    import muontomo.reconstruct as R
    monkeypatch.setattr(R, "layer_cv_score", fake_score)

    scan, best = focus.cv_height_scan(cfg, tmaps, coarse_m=0.2, fine_m=0.1)
    assert best == pytest.approx(3.0, abs=0.1)
    zs = [e["z"] for e in scan]
    assert zs == sorted(zs) and len(zs) == len(set(zs))  # sorted, deduplicated
    # the fine stage added 0.1-m points around the coarse minimum
    assert any(abs(z - 2.9) < 1e-9 or abs(z - 3.1) < 1e-9 for z in zs)
    # two-stage does fewer evaluations than a dense fine sweep of the band
    assert len(calls) < len(np.arange(2.0, 4.01, 0.1))


def test_layer_cv_score_runs_on_tiny_phantom(tiny_phantom):
    """The real (unmocked) CV scorer: two single-pose solves on a 16-bin layer."""
    from dataclasses import replace

    from muontomo.opacity import opacity_map
    from muontomo.reconstruct import layer_cv_score

    cfg, tmaps = tiny_phantom
    omaps = {p: opacity_map(t) for p, t in tmaps.items()}
    first = next(iter(tmaps.values()))
    rc2 = replace(cfg.reconstruction, algorithm="tv", tv_z_weight=0.0, n_iter=30)
    cv_mean, cv = layer_cv_score(cfg.geometry, first.txedges, first.tyedges,
                                 omaps, rc2, 3.1, 0.3, cache_dir=None,
                                 cv_trim_pct=90.0)
    assert np.isfinite(cv_mean) and cv_mean > 0
    assert set(cv) == {"pos0->pos1", "pos1->pos0"}

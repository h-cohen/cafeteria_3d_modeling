"""Single-detector refocusing: the DAQ shear correction and its consequences.

Covers the bar-coordinate origin removal in prepare_angular_hist (which otherwise
translates the whole reconstructed scene) and the joint triangulation least-squares.
"""

import copy

import numpy as np
import pytest

from muontomo import phantom
from muontomo.beams import _match_peaks, beam_peaks_subbin, triangulate
from muontomo.calibration import (
    prepare_angular_hist,
    refocus_height_m,
    transmission_maps,
)
from muontomo.config import BinningConfig, PoseConfig
from muontomo.io import Hist2D, load_dataset


def _hist(n=80, t=2.0):
    e = np.linspace(-t, t, n + 1)
    return Hist2D(values=np.ones((n, n)), xedges=e, yedges=e, name="h")


def test_refocus_height_lookup():
    assert refocus_height_m("XY07m") == 7.0
    assert refocus_height_m("XY01m") == 1.0
    assert refocus_height_m("txty") is None


def test_txty_is_not_shifted():
    """A non-refocused histogram must pass through untouched even if the origin is set."""
    h = _hist()
    b = BinningConfig(hist="txty", t_max=1.0, rebin=8, refocus_origin_m=0.19)
    out = prepare_angular_hist(h, b)
    ref = prepare_angular_hist(h, BinningConfig(hist="txty", t_max=1.0, rebin=8))
    assert np.allclose(out.xedges, ref.xedges)


def test_refocus_shift_relabels_content_not_counts():
    """The origin correction re-labels each count's angle by -b0/z, exactly.

    The crop re-anchors the edge array to +-t_max either way, so the correction is
    visible in WHERE a feature lands, not in the edge values. A feature at tan t in the
    raw map must come out at t - b0/z. Counts are only ever summed in whole bins.
    """
    z, b0 = 7.0, 0.7  # 0.1 tan units = exactly 2 bins of this 0.05-wide test grid
    h = _hist()
    h.values[:] = 0.0
    hot = 52  # raw bin whose centre sits at tan = +0.625
    h.values[hot, hot] = 1000.0
    t_hot = float(h.xcenters[hot])

    b = BinningConfig(hist="XY07m", t_max=1.0, rebin=1, refocus_origin_m=b0)
    out = prepare_angular_hist(h, b)
    assert out.values.sum() == 1000.0  # nothing lost, nothing interpolated
    i = int(np.argmax(out.values.sum(axis=1)))
    assert float(out.xcenters[i]) == pytest.approx(t_hot - b0 / z, abs=1e-9)

    plain = prepare_angular_hist(h, BinningConfig(hist="XY07m", t_max=1.0, rebin=1))
    j = int(np.argmax(plain.values.sum(axis=1)))
    assert float(plain.xcenters[j]) == pytest.approx(t_hot, abs=1e-9)


def test_shifted_grid_still_rebins():
    """A shifted edge grid no longer divides evenly by rebin; it must be trimmed, not raise."""
    h = _hist(n=800, t=2.0)
    out = prepare_angular_hist(
        h, BinningConfig(hist="XY07m", t_max=1.0, rebin=8, refocus_origin_m=0.1912)
    )
    assert out.values.shape[0] == out.values.shape[1]
    assert out.values.shape[0] > 0


def test_subbin_peaks_beat_bin_centres():
    """A peak deliberately placed off-centre is recovered to well under one bin."""
    grid = np.arange(0.0, 10.0, 1.0)
    true = 4.3
    prof = np.exp(-0.5 * ((grid - true) / 1.2) ** 2)
    got = beam_peaks_subbin(grid, prof, prom_sigmas=0.1)
    assert len(got) == 1
    assert abs(got[0] - true) < 0.25  # bin width is 1.0
    assert abs(got[0] - 4.0) > 0.15  # genuinely off the bin centre


def test_match_peaks_pairs_by_world_position():
    poses = {"pos0": PoseConfig(x=0.0), "pos1": PoseConfig(x=2.0)}
    z = 7.0
    # one beam at world x = 3.5, seen at the angle each pose must report
    peaks = {"pos0": np.array([3.5 / z]), "pos1": np.array([(3.5 - 2.0) / z])}
    feats = _match_peaks(peaks, poses, "x", z, tol_m=0.6)
    assert len(feats) == 1
    assert feats[0]["world0"] == pytest.approx(3.5, abs=1e-9)
    assert set(feats[0]["obs"]) == {"pos0", "pos1"}


def test_match_peaks_drops_unmatched():
    """A peak only one pose sees is not a triangulable feature."""
    poses = {"pos0": PoseConfig(x=0.0), "pos1": PoseConfig(x=2.0)}
    peaks = {"pos0": np.array([0.5]), "pos1": np.array([-0.5])}  # ~7 m apart in world
    assert _match_peaks(peaks, poses, "x", 7.0, tol_m=0.6) == []


@pytest.fixture(scope="module")
def beam_phantom(tmp_path_factory):
    spec = phantom.make_spec("beams")
    spec["geometry"]["pose_error"] = {}  # triangulation is tested against TRUE poses
    out = phantom.generate(spec, tmp_path_factory.mktemp("tri") / "p")
    cfg = phantom.phantom_run_config(out)
    return cfg, spec["slab"]["z0"]


def test_triangulation_recovers_phantom_ceiling(beam_phantom):
    cfg, z_true = beam_phantom
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    bin_t = float(next(iter(tmaps.values())).txedges[1] - next(iter(tmaps.values())).txedges[0])
    r = triangulate(tmaps, cfg.geometry, z_true, sigma_t=bin_t / np.sqrt(12.0))
    assert r["ok"], r.get("reason")
    assert r["n_features_x"] >= 2
    assert abs(r["z_m"] - z_true) < 0.6
    assert r["z_sigma_m"] > 0
    assert r["dof"] >= 1


def test_scale_closure_recovers_phantom_pitch(beam_phantom):
    """The (baseline, height, pitch) triple must close on a phantom of known pitch."""
    from muontomo.beams import scale_closure

    cfg, z_true = beam_phantom
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    sc = scale_closure(tmaps, cfg.geometry, z_true)
    assert sc["angular_beam_period_tan"] > 0
    # phantom "beams" preset has a 0.60 m pitch
    assert abs(sc["implied_beam_pitch_m"] - 0.60) < 0.15
    # the reported ratios must reproduce the absolute numbers
    assert sc["z_per_baseline"] * sc["baseline_m"] == pytest.approx(z_true, rel=1e-3)
    assert (sc["pitch_per_baseline"] * sc["baseline_m"]
            == pytest.approx(sc["implied_beam_pitch_m"], rel=1e-2))


def test_scale_closure_is_linear_in_baseline(beam_phantom):
    """Height and pitch both scale with the assumed baseline -- the degeneracy is real."""
    import dataclasses

    from muontomo.beams import scale_closure

    cfg, z_true = beam_phantom
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    a = scale_closure(tmaps, cfg.geometry, z_true)
    g2 = copy.deepcopy(cfg.geometry)
    pid = [p for p in tmaps if p != "pos0"][0]
    base = g2.pose(pid)
    g2.poses[pid] = dataclasses.replace(base, x=base.x * 2.0, y=base.y * 2.0)
    b = scale_closure(tmaps, g2, z_true)
    # doubling the baseline halves z-per-baseline; the angular period is untouched
    assert b["baseline_m"] == pytest.approx(2 * a["baseline_m"], rel=1e-6)
    assert b["angular_beam_period_tan"] == pytest.approx(a["angular_beam_period_tan"])
    assert b["z_per_baseline"] == pytest.approx(a["z_per_baseline"] / 2, rel=1e-3)


def test_spectral_focus_is_scene_agnostic_and_noise_corrected(beam_phantom):
    """signal_band finds the scene's own frequencies; spectral_focus scores without them."""
    from muontomo.focus import signal_band, spectral_focus

    cfg, _ = beam_phantom
    tmaps = transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)
    tm = tmaps["pos0"]
    band = signal_band(tm)
    assert band is not None
    lo, hi = band
    assert 0.0 <= lo <= hi <= 0.5

    score = spectral_focus(tm, band)
    assert np.isfinite(score) and score >= 0.0
    # blurring the map must lower the score -- that is what makes it a focus metric
    from scipy.ndimage import gaussian_filter
    import dataclasses

    blurred = dataclasses.replace(tm, T=gaussian_filter(tm.T, 1.5))
    assert spectral_focus(blurred, band) < score


def test_spectral_focus_rejects_pure_noise():
    """With no structure above the Poisson floor there is no signal band to find."""
    from muontomo.calibration import TransmissionMap
    from muontomo.focus import signal_band

    rng = np.random.default_rng(0)
    n = 64
    e = np.linspace(-1, 1, n + 1)
    T = np.clip(1.0 + 0.05 * rng.standard_normal((n, n)), 0.2, None)
    tm = TransmissionMap(T=T, sigma_T=np.full((n, n), 0.05),
                         mask=np.ones((n, n), bool), txedges=e, tyedges=e,
                         n_cafe=np.full((n, n), 400.0), n_sky=np.full((n, n), 400.0),
                         scale=1.0, pose_id="p")
    assert signal_band(tm, snr_min=3.0) is None


def test_triangulation_reports_underdetermined():
    """With a single pose there is no parallax and the fit must decline, not guess."""
    poses = {"pos0": PoseConfig(x=0.0)}

    class G:
        def pose(self, pid):
            return poses[pid]

    class TM:
        txedges = tyedges = np.linspace(-1, 1, 51)
        mask = np.ones((50, 50), dtype=bool)
        T = np.full((50, 50), 0.9)
        txcenters = tycenters = 0.5 * (txedges[:-1] + txedges[1:])

    r = triangulate({"pos0": TM()}, G(), 7.0, sigma_t=0.01)
    assert r["ok"] is False

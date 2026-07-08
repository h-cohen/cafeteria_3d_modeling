"""Unit tests for io/calibration/opacity/geometry against hand-computed fixtures."""

import numpy as np
import pytest

from muontomo.calibration import compute_transmission, estimate_scale
from muontomo.config import CalibrationConfig, PoseConfig, RunConfig
from muontomo.geometry import VoxelGrid, aperture_offsets, bin_directions
from muontomo.io import Hist2D
from muontomo.opacity import opacity_map


def _hist(values, lo=-1.0, hi=1.0):
    values = np.asarray(values, dtype=float)
    return Hist2D(
        values=values,
        xedges=np.linspace(lo, hi, values.shape[0] + 1),
        yedges=np.linspace(lo, hi, values.shape[1] + 1),
    )


class TestHist2D:
    def test_rebin_sums_counts(self):
        h = _hist(np.arange(16).reshape(4, 4))
        r = h.rebin(2)
        assert r.values.shape == (2, 2)
        assert r.values[0, 0] == 0 + 1 + 4 + 5
        assert r.values.sum() == h.values.sum()

    def test_crop_keeps_inner_window(self):
        h = _hist(np.ones((8, 8)), lo=-2, hi=2)
        c = h.crop((-1, 1), (-1, 1))
        assert c.values.shape == (4, 4)
        assert c.xedges[0] == -1 and c.xedges[-1] == 1

    def test_rebin_rejects_indivisible(self):
        with pytest.raises(ValueError):
            _hist(np.ones((5, 5))).rebin(2)


class TestTransmission:
    def test_open_sky_gives_unit_transmission(self):
        # cafe = sky/10 exactly -> after scale normalization T == 1 everywhere
        sky = _hist(np.full((10, 10), 1000.0))
        cafe = _hist(np.full((10, 10), 100.0))
        cal = CalibrationConfig(norm_smooth_bins=0)
        t = compute_transmission(cafe, sky, cal)
        assert np.allclose(t.T[t.mask], 1.0)
        assert np.isclose(t.scale, 0.1)

    def test_absorber_reduces_transmission(self):
        sky_counts = np.full((10, 10), 10000.0)
        cafe_counts = np.full((10, 10), 1000.0)
        cafe_counts[4:6, :] = 500.0  # a "beam" halving the flux
        t = compute_transmission(_hist(cafe_counts), _hist(sky_counts), CalibrationConfig(norm_smooth_bins=0))
        assert np.allclose(t.T[4:6, :], 0.5, atol=0.01)
        lam = opacity_map(t)
        assert np.allclose(lam.lam[4:6, :], np.log(2), atol=0.02)
        assert np.allclose(lam.lam[0, :], 0.0, atol=0.01)

    def test_mask_kills_low_stat_bins(self):
        sky_counts = np.full((4, 4), 1000.0)
        sky_counts[0, 0] = 3.0
        cafe_counts = np.full((4, 4), 100.0)
        cafe_counts[1, 1] = 1.0
        t = compute_transmission(_hist(cafe_counts), _hist(sky_counts), CalibrationConfig(norm_smooth_bins=0))
        assert not t.mask[0, 0] and not t.mask[1, 1]
        assert t.mask.sum() == 14

    def test_sigma_matches_poisson(self):
        sky = _hist(np.full((4, 4), 400.0))
        cafe = _hist(np.full((4, 4), 100.0))
        t = compute_transmission(cafe, sky, CalibrationConfig(norm_smooth_bins=0))
        expect = 1.0 * np.sqrt(1 / 100 + 1 / 400)
        assert np.allclose(t.sigma_T[t.mask], expect)

    def test_scale_uses_quantile_not_total(self):
        # Half the map is behind an absorber; the scale must come from the open half.
        sky = np.full((10, 10), 10000.0)
        cafe = np.full((10, 10), 1000.0)
        cafe[:5] = 200.0
        s = estimate_scale(cafe, sky, CalibrationConfig(norm_smooth_bins=0, norm_quantile=0.9))
        assert np.isclose(s, 0.1, rtol=0.01)


class TestGeometry:
    def test_vertical_bin_direction(self):
        d = bin_directions(np.array([0.0]), np.array([0.0]), PoseConfig())
        assert np.allclose(d[0, 0], [0, 0, 1])

    def test_yaw_rotates_direction(self):
        d = bin_directions(np.array([0.5]), np.array([0.0]), PoseConfig(yaw_deg=90.0))
        v = d[0, 0] * np.sqrt(1 + 0.25)  # un-normalize
        assert np.allclose(v, [0.0, 0.5, 1.0], atol=1e-12)

    def test_aperture_offsets_span_aperture(self):
        offs = aperture_offsets(PoseConfig(), aperture_m=0.65, n_sub=4)
        assert offs.shape == (16, 3)
        assert np.allclose(offs[:, 2], 0)
        assert np.isclose(offs[:, 0].max(), 0.65 * 3 / 8)
        assert np.isclose(offs.mean(axis=0)[0], 0.0)

    def test_grid_axes(self):
        g = VoxelGrid(origin=(0.0, 0.0, 1.0), spacing=0.5, shape=(2, 2, 4))
        assert np.allclose(g.axis_centers(2), [1.25, 1.75, 2.25, 2.75])
        assert g.extent(2) == (1.0, 3.0)
        assert g.n_voxels == 16


class TestConfig:
    def test_json_roundtrip(self, tmp_path):
        cfg = RunConfig()
        cfg.geometry.poses["pos1"] = PoseConfig(x=1.7, yaw_deg=2.0)
        p = tmp_path / "c.json"
        cfg.save(p)
        back = RunConfig.load(p)
        assert back.to_dict() == cfg.to_dict()
        assert back.hash() == cfg.hash()

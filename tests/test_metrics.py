"""Each metric on tiny arrays with known answers."""

import numpy as np

from muontomo.metrics import crossval as cv
from muontomo.metrics import fidelity, structure
from muontomo.metrics.truth import iou_dense, rmse_scaled, ssim3d, z_error_m
from muontomo.metrics.volume import volume_stats


class TestFidelity:
    def test_chi2_zero_for_perfect_prediction(self):
        lam = np.random.default_rng(0).normal(1, 0.1, (8, 8))
        assert fidelity.chi2_ndof(lam, lam, np.ones_like(lam)) == 0.0

    def test_chi2_is_one_for_1sigma_residuals(self):
        lam = np.zeros((10, 10))
        pred = np.full((10, 10), 0.5)
        w = np.full((10, 10), 1 / 0.25)  # sigma = 0.5
        assert np.isclose(fidelity.chi2_ndof(lam, pred, w), 1.0)

    def test_deviance_matches_chi2_at_high_counts(self):
        rng = np.random.default_rng(1)
        mu = np.full((20, 20), 10000.0)
        n = rng.poisson(mu).astype(float)
        dev = fidelity.deviance_ndof(n, mu, np.ones(mu.shape, bool))
        assert 0.8 < dev < 1.2

    def test_aligned_chi2_finds_shift(self):
        rng = np.random.default_rng(2)
        lam = rng.normal(0, 1, (16, 16))
        pred = np.roll(lam, (2, -1), axis=(0, 1))
        best, shift = fidelity.chi2_aligned(lam, pred, np.ones_like(lam))
        assert shift == (-2, 1)
        assert best < 1e-12


class TestCrossval:
    def test_free_offset_absorbs_constant(self):
        lam = np.full((6, 6), 2.0)
        pred = np.zeros((6, 6))
        assert np.isclose(cv.heldout_chi2(lam, pred, np.ones_like(lam)), 0.0)

    def test_pearson_perfect_correlation(self):
        rng = np.random.default_rng(3)
        lam = rng.normal(0, 1, (8, 8))
        assert np.isclose(cv.heldout_pearson(lam, 3 * lam + 1, np.ones_like(lam)), 1.0)


class TestStructure:
    def test_periodicity_detects_stripes(self):
        x = np.arange(64)
        img = np.sin(2 * np.pi * x / 8)[:, None] * np.ones((1, 64))
        per = structure.periodicity(img)
        assert per["snr"] > 10
        assert 7 < per["period_bins"] < 9
        flat = structure.periodicity(np.random.default_rng(0).normal(size=(64, 64)))
        assert per["snr"] > 3 * flat["snr"]

    def test_stripe_contrast(self):
        img = np.ones((32, 32))
        img[::4] = 0.5  # dark stripes
        st = structure.stripe_stats(img)
        assert np.isclose(st["contrast"], (1 - 0.5) / 1.5, atol=0.02)

    def test_flat_noise_scales_with_sigma(self):
        rng = np.random.default_rng(4)
        lo = structure.flat_noise(rng.normal(0, 0.1, (64, 64)))
        hi = structure.flat_noise(rng.normal(0, 1.0, (64, 64)))
        assert 5 < hi / lo < 15


class TestVolume:
    def test_thin_slab_has_small_z_width(self):
        rho = np.zeros((10, 10, 20))
        rho[:, :, 10] = 1.0
        zc = np.linspace(0.05, 1.95, 20)
        st = volume_stats(rho, zc)
        assert np.isclose(st["z_peak_m"], zc[10])
        assert st["z_eff_width_m"] < 0.15
        assert st["neg_mass_frac"] == 0.0

    def test_negative_mass_fraction(self):
        rho = np.ones((4, 4, 4))
        rho[0, 0, 0] = -32.0
        st = volume_stats(rho, np.arange(4.0))
        assert np.isclose(st["neg_mass_frac"], 32 / (63 + 32))


class TestTruth:
    def test_rmse_scale_invariant(self):
        tru = np.random.default_rng(5).uniform(0, 1, (8, 8, 8))
        assert rmse_scaled(2.5 * tru, tru) < 1e-12

    def test_ssim_identity(self):
        tru = np.random.default_rng(6).uniform(0, 1, (12, 12, 12))
        assert ssim3d(tru, tru) > 0.99
        assert ssim3d(np.roll(tru, 3, axis=0), tru) < 0.6

    def test_iou_identical_masks(self):
        tru = np.zeros((8, 8, 8))
        tru[2:5, 2:5, 2:5] = 1.0
        assert iou_dense(tru, tru) == 1.0
        assert iou_dense(np.roll(tru, 3, axis=0), tru) < 0.5

    def test_z_error(self):
        a = np.zeros((4, 4, 10))
        b = np.zeros((4, 4, 10))
        a[:, :, 2] = 1
        b[:, :, 6] = 1
        zc = np.linspace(0, 9, 10)
        assert np.isclose(z_error_m(a, b, zc), 4.0)

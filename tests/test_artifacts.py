"""Unit tests for the artifact-cleanup enhancers (muontomo.enhance.artifacts)
and the shared verification gate, on a fully synthetic EnhanceContext -- no
real data, no torch.
"""

import json

import numpy as np
import pytest

from muontomo.config import RunConfig
from muontomo.enhance import artifacts
from muontomo.enhance.context import EnhanceContext
from muontomo.enhance.verify import verify

BEAMS = [-3.12, -1.54, 0.14, 1.64, 3.32]


def _stripes(xs, ys, amp=1.0, width=0.2):
    img = np.zeros((xs.size, ys.size))
    for b in BEAMS:
        img += amp * np.exp(-0.5 * ((xs[:, None] - b) / width) ** 2)
    return img


@pytest.fixture()
def ctx(tmp_path):
    """Synthetic context: noisy striped ceiling layer + bright corner artifacts,
    a clean measured guide, and a metrics.json carrying the verified beams."""
    rng = np.random.default_rng(0)
    res = 0.1
    xs = np.arange(-5 + res / 2, 5, res)
    ys = np.arange(-5 + res / 2, 5, res)
    clean = _stripes(xs, ys)
    layer = clean + 0.15 * rng.standard_normal(clean.shape)
    # limited-angle style corner blobs: brighter than the beams, on the rim
    layer[xs > 4.3, :] += 3.0
    layer[:, ys < -4.3] += 3.0
    layer = np.maximum(layer, 0.0)
    guide = clean + 0.03 * rng.standard_normal(clean.shape)

    (tmp_path / "metrics.json").write_text(
        json.dumps({"beams": {"beams_x_data_m": BEAMS}}))
    cfg = RunConfig()
    cfg.geometry.grid_xy_m = ((-5.0, 5.0), (-5.0, 5.0))
    return EnhanceContext(
        run=tmp_path, cfg=cfg, tmaps={}, omaps={},
        rho=np.zeros((1, 1, 1)), origin=(-5.0, -5.0, 1.0), spacing=res,
        z_layer=7.0, iz=0, xs=xs, ys=ys, layer=layer,
        guide=np.maximum(guide, 0.0),
        per_pose_guide={"pos0": np.maximum(guide, 0.0)}, sharp_guide_id="pos0",
    )


def test_coverage_mask_keeps_beams_kills_rim(ctx):
    m = artifacts.coverage_mask(ctx)
    assert m.shape == ctx.layer.shape
    ic = np.argmin(np.abs(ctx.xs - 0.14))
    jc = np.argmin(np.abs(ctx.ys))
    assert m[ic, jc] > 0.95  # centre beam fully kept
    assert m[-1, jc] < 0.05  # x rim (beyond beams + margin) suppressed
    assert m[ic, 0] < 0.05  # y rim suppressed


def test_floor_subtract_lowers_interbeam_pedestal(ctx):
    img = ctx.layer + 0.3  # add a uniform pedestal
    mask = artifacts.coverage_mask(ctx)
    out = artifacts._floor_subtract(ctx, img, mask)
    far = np.abs(ctx.xs[:, None] - np.asarray(BEAMS)[None, :]).min(axis=1) > 0.5
    band = (ctx.ys > -2) & (ctx.ys < 2)
    assert out[np.ix_(far, band)].mean() < img[np.ix_(far, band)].mean()
    assert (out >= 0).all()


def test_directional_sharpen_nonneg_and_peak_preserving(ctx):
    smooth = artifacts.gaussian_filter(ctx.guide, 2.0)
    out = artifacts.directional_sharpen(ctx, smooth)
    assert (out >= 0).all()
    prof_in = smooth[:, (ctx.ys > -2) & (ctx.ys < 2)].mean(axis=1)
    prof_out = out[:, (ctx.ys > -2) & (ctx.ys < 2)].mean(axis=1)
    i = np.argmin(np.abs(ctx.xs - 0.14))
    assert prof_out[i] >= prof_in[i] * 0.9  # beam peak not destroyed


def test_clean_enhancer_improves_cnr_and_passes_gate(ctx):
    enh = artifacts._Clean()
    out = enh.enhance(ctx)
    assert out.shape == ctx.layer.shape and (out >= 0).all()
    info = enh.last_info
    assert info["cnr_after"] > info["cnr_before"]
    metrics = verify(ctx, out)
    assert metrics["verdict"] == "PASS"
    assert metrics["n_beams"] == len(BEAMS)
    assert metrics["mean_abs_beam_offset_m"] <= 0.15


def test_dipclean_uses_precomputed_dip_layer(ctx):
    """dipclean loads enhance/dip.npy when present -- no torch needed -- and
    must keep the beams while killing the rim artifacts."""
    rng = np.random.default_rng(3)
    dip_layer = _stripes(ctx.xs, ctx.ys) + 0.05 * rng.standard_normal(
        (ctx.xs.size, ctx.ys.size))
    dip_layer[ctx.xs > 4.3, :] += 2.0  # residual rim blob DIP could not remove
    (ctx.run / "enhance").mkdir()
    np.save(ctx.run / "enhance" / "dip.npy", np.maximum(dip_layer, 0.0))

    enh = artifacts._DipClean()
    out = enh.enhance(ctx)
    assert enh.last_info["cnr_after"] > enh.last_info["cnr_before"]
    # rim blob strongly suppressed (taper reaches exact zero at ~4.85 m; before
    # that it rolls off smoothly), and fully zero past the roll
    rim_in = np.maximum(dip_layer, 0.0)[ctx.xs > 4.5, :].max()
    assert out[ctx.xs > 4.5, :].max() < 0.2 * rim_in
    assert out[ctx.xs > 4.85, :].max() == pytest.approx(0.0, abs=1e-9)
    metrics = verify(ctx, out)
    assert metrics["verdict"] == "PASS" and metrics["n_beams"] == len(BEAMS)


def test_registry_exposes_both_enhancers():
    from muontomo.enhance import REGISTRY

    assert "clean" in REGISTRY and "dipclean" in REGISTRY

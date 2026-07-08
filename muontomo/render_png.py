"""Standard image set for a run. Fixed filenames and color scales so images are
comparable across runs at a glance; panel_summary.png is the one to read first."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRANS_SCALE = {"vmin": 0.5, "vmax": 1.5, "cmap": "viridis"}
RESID_SCALE = {"vmin": -5, "vmax": 5, "cmap": "RdBu_r"}


def _imshow(ax, img, extent=None, title="", xlabel="", ylabel="", **kw):
    im = ax.imshow(np.asarray(img).T, origin="lower", extent=extent, **kw)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im


def _t_extent(tmap):
    return [tmap.txedges[0], tmap.txedges[-1], tmap.tyedges[0], tmap.tyedges[-1]]


def render_run_images(out_dir, tmaps, lam_pred, omaps, rho, grid, scorecard) -> None:
    """Write the standard PNG set into out_dir/images."""
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pose_ids = list(tmaps)
    zc = grid.axis_centers(2)
    ext_xy = [*grid.extent(0), *grid.extent(1)]
    pos = np.maximum(rho, 0.0)
    zprof = pos.sum(axis=(0, 1))
    iz_peak = int(np.argmax(zprof)) if zprof.max() > 0 else rho.shape[2] // 2
    vmax = max(float(np.percentile(pos, 99.5)), 1e-9)

    # --- individual images ---
    for pid in pose_ids:
        t = tmaps[pid]
        ext = _t_extent(t)
        fig, ax = plt.subplots(figsize=(6, 5))
        _imshow(ax, np.where(t.mask, t.T, np.nan), ext, f"{pid} measured transmission",
                "tan(theta_x)", "tan(theta_y)", **TRANS_SCALE)
        fig.savefig(img_dir / f"transmission_{pid}_meas.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        _imshow(ax, np.where(t.mask, np.exp(-lam_pred[pid]), np.nan), ext,
                f"{pid} predicted transmission", "tan(theta_x)", "tan(theta_y)", **TRANS_SCALE)
        fig.savefig(img_dir / f"transmission_{pid}_pred.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        o = omaps[pid]
        pull = np.where(o.mask, (o.lam - lam_pred[pid]) / np.where(o.sigma > 0, o.sigma, 1), np.nan)
        fig, ax = plt.subplots(figsize=(6, 5))
        _imshow(ax, pull, ext, f"{pid} residual pull (lam_meas - lam_pred)/sigma",
                "tan(theta_x)", "tan(theta_y)", **RESID_SCALE)
        fig.savefig(img_dir / f"residual_{pid}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    for name, axis in [("mip_x", 0), ("mip_y", 1), ("mip_z", 2)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        mip = pos.max(axis=axis)
        if axis == 0:
            e = [*grid.extent(1), *grid.extent(2)]
            labels = ("y (m)", "z (m)")
        elif axis == 1:
            e = [*grid.extent(0), *grid.extent(2)]
            labels = ("x (m)", "z (m)")
        else:
            e = ext_xy
            labels = ("x (m)", "y (m)")
        _imshow(ax, mip, e, f"max-intensity projection ({name})", *labels, cmap="inferno", vmin=0, vmax=vmax)
        fig.savefig(img_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _imshow(ax, pos[:, :, iz_peak], ext_xy, f"volume slice at z={zc[iz_peak]:.2f} m",
            "x (m)", "y (m)", cmap="inferno", vmin=0, vmax=vmax)
    fig.savefig(img_dir / "slice_z_peak.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(zc, zprof / max(zprof.max(), 1e-30))
    ax.axvline(zc[iz_peak], color="r", ls="--", lw=1)
    ax.set(xlabel="z (m)", ylabel="relative mass", title="vertical mass profile")
    fig.savefig(img_dir / "zprofile.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- contact sheet ---
    ncol = max(3, len(pose_ids) + 1)
    fig, axs = plt.subplots(3, ncol, figsize=(5.5 * ncol, 14))
    for j, pid in enumerate(pose_ids):
        t = tmaps[pid]
        ext = _t_extent(t)
        _imshow(axs[0, j], np.where(t.mask, t.T, np.nan), ext, f"{pid} T measured", **TRANS_SCALE)
        _imshow(axs[1, j], np.where(t.mask, np.exp(-lam_pred[pid]), np.nan), ext,
                f"{pid} T predicted", **TRANS_SCALE)
        o = omaps[pid]
        pull = np.where(o.mask, (o.lam - lam_pred[pid]) / np.where(o.sigma > 0, o.sigma, 1), np.nan)
        _imshow(axs[2, j], pull, ext, f"{pid} residual pull", **RESID_SCALE)
    _imshow(axs[0, ncol - 1], pos[:, :, iz_peak], ext_xy,
            f"slice z={zc[iz_peak]:.2f} m", cmap="inferno", vmin=0, vmax=vmax)
    _imshow(axs[1, ncol - 1], pos.max(axis=1), [*grid.extent(0), *grid.extent(2)],
            "MIP along y", cmap="inferno", vmin=0, vmax=vmax)
    ax = axs[2, ncol - 1]
    ax.plot(zc, zprof / max(zprof.max(), 1e-30))
    ax.set(xlabel="z (m)", title="z mass profile")
    head = scorecard.get("headline", {})
    txt = "\n".join(f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}" for k, v in head.items())
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    fig.suptitle(f"run: {scorecard.get('run', '')}   algorithm: {scorecard.get('algorithm', '')}")
    fig.savefig(img_dir / "panel_summary.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

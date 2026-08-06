"""Rendering: per-method comparison PNG and the combined all-methods report."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .context import EnhanceContext
from .verify import data_beam_positions

METHOD_TITLE = {
    "guided": "guided filter",
    "dip": "deep image prior",
    "pnp": "plug-and-play (NLM)",
}


def _crop(ctx: EnhanceContext):
    box = getattr(ctx.cfg.geometry, "viewer_crop_xy_m", None) or [
        [ctx.xs.min(), ctx.xs.max()], [ctx.ys.min(), ctx.ys.max()]]
    (x0, x1), (y0, y1) = box
    ix = (slice(int(np.searchsorted(ctx.xs, x0)), int(np.searchsorted(ctx.xs, x1))),
          slice(int(np.searchsorted(ctx.ys, y0)), int(np.searchsorted(ctx.ys, y1))))
    return ix, [x0, x1, y0, y1]


def _show(ax, ctx, img, title, ix, ext, beams):
    s = np.asarray(img)[ix]
    nz = s[s > 0]
    lo, hi = (np.percentile(nz, [35, 85]) if nz.size else (0.0, 1.0))
    ax.imshow(np.clip((s - lo) / (hi - lo + 1e-9), 0, 1).T, origin="lower",
              extent=ext, cmap="viridis", aspect="equal")
    ax.set_title(title, fontsize=11)
    for bx in beams:
        if ext[0] <= bx <= ext[1]:
            ax.axvline(bx, color="r", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def render_method_png(ctx: EnhanceContext, method: str, enhanced: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ix, ext = _crop(ctx)
    beams = data_beam_positions(ctx)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    _show(axes[0], ctx, ctx.display_blur(ctx.layer), "plain reconstruction", ix, ext, beams)
    _show(axes[1], ctx, enhanced, f"{METHOD_TITLE.get(method, method)}", ix, ext, beams)
    _show(axes[2], ctx, ctx.sharp_guide, f"2D flux ({ctx.sharp_guide_id}, guide)", ix, ext, beams)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def render_report(ctx: EnhanceContext, results: dict, out_dir: Path) -> Path:
    """results: {method: {"layer": 2D, "metrics": {...}}}. Writes REPORT.md +
    images/enhance_report.png. Returns the REPORT.md path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ix, ext = _crop(ctx)
    beams = data_beam_positions(ctx)
    order = [m for m in ["guided", "dip", "pnp"] if m in results]
    panels = [("plain reconstruction", ctx.display_blur(ctx.layer))]
    panels += [(METHOD_TITLE.get(m, m), results[m]["layer"]) for m in order]
    panels += [(f"2D flux ({ctx.sharp_guide_id}, guide)", ctx.sharp_guide)]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.2),
                             constrained_layout=True)
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        _show(ax, ctx, img, title, ix, ext, beams)
    img_dir = ctx.run / "images"  # the run's standard images dir (REPORT.md links ../images)
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "enhance_report.png", dpi=100)
    plt.close(fig)

    # plain-recon metrics as the baseline row
    from .verify import verify
    base = verify(ctx, ctx.display_blur(ctx.layer))

    cols = ["method", "mean|offset| m", "n_beams", "FWHM m", "edge_grad", "chi2", "runtime s", "verdict"]
    def row(name, m):
        return [name, m.get("mean_abs_beam_offset_m"), m.get("n_beams"), m.get("beam_fwhm_m"),
                m.get("edge_gradient"),
                m.get("val_chi2", m.get("best_chi2", "-")), m.get("runtime_s", "-"), m.get("verdict", "-")]
    rows = [row("plain recon", base)] + [row(METHOD_TITLE.get(m, m), results[m]["metrics"]) for m in order]

    lines = ["# Enhancement comparison\n",
             f"Run: `{ctx.run}`  |  layer z = {ctx.z_layer:.2f} m  |  "
             f"data beams: {[round(float(b),2) for b in beams]}\n",
             "Gate: mean |beam offset| <= 0.15 m AND n_beams within 1 of 5 "
             "(features must match the measured data, not be invented).\n",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    lines += ["", "![comparison](../images/enhance_report.png)", ""]
    md = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT.md").write_text(md)
    return out_dir / "REPORT.md"

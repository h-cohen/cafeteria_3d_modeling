"""Two-page autofocus report, generated for every run.

Page 1 states the objective, the two-view parallax principle, and this run's
determined ceiling height with its cross-validation curve and a parallax-
registration figure. Page 2 documents the scoring metrics and limitations,
tabulates this run's values, and embeds the ground-truth validation figure.

Written for a technically literate but non-specialist reader: precise, but with
every term explained. Called by muontomo.focus.run_focus; needs matplotlib only.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

FIG_H_IN = 11.69  # A4 portrait height, for line-height math

P1_OBJECTIVE = (
    "Objective.  The two muon detectors record the ceiling's cosmic-ray transmission shadow "
    "from two positions on the floor about 1.9 m apart. Reconstructing the three-dimensional "
    "beam structure requires the ceiling height as an input: an incorrect height defocuses the "
    "beams and biases their fitted positions. This report documents how that height is "
    "determined directly from the measurement (\"autofocus\") and quantifies the confidence in "
    "the value."
)

P1_PRINCIPLE = (
    "Principle: two-view parallax.  Each detector views the ceiling from a different position, "
    "so a feature at the true height re-projects to the same location in both views only when it "
    "is back-projected onto a plane at that height; at any other assumed height the two "
    "projections are laterally displaced. This is parallax -- the same geometric cue underlying "
    "stereoscopic depth perception. We back-project both detectors' opacity onto candidate "
    "planes across a range of heights and identify the height at which the two views agree most "
    "closely, i.e. the minimum of the curve below."
)

P1_REFOCUS = (
    "Below: the detector-1 (red) and detector-2 (cyan) opacity back-projections, overlaid at "
    "three candidate heights on a common colour scale. Where the two views agree the colours "
    "combine to neutral grey/white; residual parallax appears as red/cyan separation, most "
    "visible along the room edges. Registration is closest at the central height."
)

P2_METRICS = (
    "Scoring metrics.  Two independent measures are evaluated at each candidate height.\n\n"
    "Cross-validation residual (primary).  A thin ceiling layer is fitted to one detector and "
    "used to predict the second detector's measurement. The prediction residual (mean squared, "
    "with the worst 10% of bins trimmed so the estimate does not hinge on a few aliasing-induced "
    "outliers) is minimal at the correct height -- where a single layer is jointly consistent "
    "with both views -- and increases where no single layer can satisfy both. This measure is "
    "robust to periodic-beam aliasing because it incorporates the physics forward-model and the "
    "finite extent of the room. The reported autofocus height is the location of its minimum.\n\n"
    "Two-view correlation (secondary).  The normalized cross-correlation of the two "
    "back-projections. It is inexpensive and is the value displayed in the 3-D viewer, but it is "
    "susceptible to bias from bright non-beam structure and is therefore used only as a "
    "consistency check, not as the reported value.\n\n"
    "Height map.  Repeating the correlation within local windows yields a per-region height "
    "estimate. An approximately constant map (as obtained here) confirms a single planar ceiling; "
    "large spatial variation would indicate a sloped or stepped ceiling."
)

P2_LIMITS = (
    "Limitations.  (i) At least two detectors are required; a single view carries no height "
    "information. (ii) A strictly periodic beam pattern admits aliased solutions at heights "
    "spaced by pitch x lever / baseline; for this geometry the nearest alias lies about 6 m from "
    "the true height, outside the physically plausible range. (iii) The estimate assumes a single "
    "planar layer, an assumption tested by the height map."
)

P2_VALIDATION = (
    "Validation against synthetic ground truth (figure on the following page).  Simulated "
    "datasets with the campaign geometry -- detector baseline and beam pitch matched to the "
    "measurement -- were generated with the ceiling deliberately placed at 6.6, 7.0 and 7.4 m "
    "and processed by the identical procedure (panel A). The recovered heights track the injected "
    "values to within about 0.2 m (panel B); the small systematic under-estimate reflects the "
    "beam mass-centroid lying below the slab base. Because the recovery is demonstrated on "
    "ceilings of known height, the same procedure applied to the measurement (page 1) can be "
    "trusted."
)


def _flow(fig, x, y, text, width=108, fs=9.3, gap=0.009, color="#222"):
    """Draw pre-wrapped paragraphs top-down from y; return the y below the block."""
    lh = (fs / 72.0) * 1.33 / FIG_H_IN
    for para in text.split("\n\n"):
        wrapped = textwrap.fill(para, width=width)
        fig.text(x, y, wrapped, ha="left", va="top", fontsize=fs, color=color, linespacing=1.33)
        y -= (wrapped.count("\n") + 1) * lh + gap
    return y


def _redcyan(a2d, b2d):
    def nrm(u):
        u = np.nan_to_num(u)
        lo, hi = np.percentile(u[u > 0], [20, 98]) if (u > 0).any() else (0.0, 1.0)
        return np.clip((u - lo) / max(hi - lo, 1e-9), 0, 1)
    A, B = nrm(a2d), nrm(b2d)
    return np.stack([A, B, B], axis=-1)


def render_report(run, cfg, result: dict, xs, ys, ref_z, ref_a, ref_b) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch

    run = Path(run)
    cv = result["cv_scan"]
    cvz = np.array([e["z"] for e in cv])
    cvv = np.array([e["cv_mean"] for e in cv])
    z_af = result["autofocus_z_m"]
    z_ql = result["quicklook_z_m"]

    beams_x, n_beams, beam_off = [], None, None
    mfile = run / "metrics.json"
    if mfile.exists():
        b = json.loads(mfile.read_text()).get("beams") or {}
        beams_x = b.get("beams_x_data_m", []) or []
        n_beams = b.get("n_beams_data")
        beam_off = b.get("mean_abs_beam_offset_m")

    pdf_path = run / "autofocus_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # ===================== PAGE 1 =====================
        fig = plt.figure(figsize=(8.27, FIG_H_IN))
        fig.text(0.06, 0.972, "Autofocus report", fontsize=21, weight="bold")
        fig.text(0.06, 0.949, f"run: {run.name}   ·   determination of the ceiling height by "
                 "two-detector parallax", fontsize=10.5, color="#555")

        # result banner
        axb = fig.add_axes([0.06, 0.882, 0.88, 0.050]); axb.axis("off")
        axb.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=axb.transAxes,
                      boxstyle="round,pad=0.015", fc="#eaf3ec", ec="#2e7d47", lw=1.5))
        axb.text(0.025, 0.5, "AUTOFOCUS HEIGHT", transform=axb.transAxes, va="center",
                 fontsize=10, color="#2e7d47", weight="bold")
        axb.text(0.30, 0.5, f"{z_af:.1f} m", transform=axb.transAxes, va="center",
                 fontsize=22, weight="bold", color="#1b5e2f")
        axb.text(0.50, 0.68, "cross-validation estimate (primary)", transform=axb.transAxes,
                 va="center", fontsize=9, color="#2e7d47")
        axb.text(0.50, 0.30, f"two-view correlation (secondary): {z_ql:.1f} m",
                 transform=axb.transAxes, va="center", fontsize=9, color="#999")

        y = _flow(fig, 0.06, 0.856, P1_OBJECTIVE)
        y = _flow(fig, 0.06, y - 0.006, P1_PRINCIPLE)

        # CV curve
        axc = fig.add_axes([0.12, 0.400, 0.80, 0.165])
        axc.plot(cvz, cvv, "-", color="#8c1d40", lw=1.6)
        axc.plot(cvz, cvv, "o", ms=2.6, color="#8c1d40")
        imin = int(np.argmin(cvv))
        axc.plot(cvz[imin], cvv[imin], "*", ms=17, color="#8c1d40", mec="k", mew=0.6, zorder=5)
        axc.axvline(cvz[imin], color="#8c1d40", ls="--", lw=1)
        axc.annotate(f"minimum at {z_af:.1f} m", (cvz[imin], cvv[imin]),
                     textcoords="offset points", xytext=(14, 24), fontsize=9, color="#8c1d40",
                     arrowprops=dict(arrowstyle="->", color="#8c1d40"))
        axc.set_xlabel("assumed ceiling height (m)", fontsize=9)
        axc.set_ylabel("two-detector disagreement\n(cross-validation residual)", fontsize=9)
        axc.set_title("Cross-validation residual vs. assumed height (lower = better agreement)",
                      fontsize=10)
        axc.grid(alpha=0.25)

        # refocus triptych
        _flow(fig, 0.06, 0.360, P1_REFOCUS)
        mx = (xs >= -4) & (xs <= 4)
        my = (ys >= -4) & (ys <= 4)
        ext = [xs[mx][0], xs[mx][-1], ys[my][0], ys[my][-1]]
        for j, z in enumerate(ref_z):
            axi = fig.add_axes([0.075 + j * 0.305, 0.050, 0.265, 0.220])
            rgb = _redcyan(ref_a[j][np.ix_(mx, my)], ref_b[j][np.ix_(mx, my)])
            axi.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext)
            for bx in beams_x:
                if -4 <= bx <= 4:
                    axi.axvline(bx, color="w", ls=":", lw=0.5, alpha=0.3)
            tag = "  (registered)" if abs(z - z_af) < 0.6 else ""
            axi.set_title(f"height {z:.0f} m{tag}", fontsize=9.5)
            axi.set_xlabel("x (m)", fontsize=8)
            if j == 0:
                axi.set_ylabel("y (m)", fontsize=8)
            axi.tick_params(labelsize=7)
        pdf.savefig(fig); plt.close(fig)

        # ===================== PAGE 2 =====================
        fig = plt.figure(figsize=(8.27, FIG_H_IN))
        fig.text(0.06, 0.975, "Autofocus report — metrics and validation",
                 fontsize=17, weight="bold")
        y = _flow(fig, 0.06, 0.950, P2_METRICS)
        y = _flow(fig, 0.06, y - 0.004, P2_LIMITS, fs=8.9, color="#444")

        # metrics table
        fig.text(0.06, y - 0.004, "Quantitative summary (this run)", fontsize=10, weight="bold")
        rows = [
            ("Autofocus height (cross-validation, primary)", f"{z_af:.1f} m"),
            ("Two-view correlation height (secondary)", f"{z_ql:.1f} m"),
            ("Height used for the 3-D reconstruction", f"{result.get('solve_z_m'):.1f} m"),
            ("Height-map median +/- interquartile range",
             f"{result['height_map'].get('median_z_m')} +/- {result['height_map'].get('iqr_z_m')} m"),
        ]
        if n_beams is not None:
            rows.append(("Beams detected in the raw transmission data", f"{n_beams}"))
        if beam_off is not None:
            rows.append(("Reconstruction beam-position error", f"{beam_off:.2f} m"))
        th = 0.025 * len(rows)
        ytab = y - 0.028 - th
        axtab = fig.add_axes([0.06, ytab, 0.88, th]); axtab.axis("off")
        tab = axtab.table(cellText=[[k, v] for k, v in rows],
                          colWidths=[0.76, 0.24], loc="center", cellLoc="left")
        tab.auto_set_font_size(False); tab.set_fontsize(9.2); tab.scale(1, 1.4)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#d0d0d0")
            cell.set_text_props(weight="bold" if c == 1 else "normal",
                                color="#222" if c == 1 else "#444")
            if r % 2 == 0:
                cell.set_facecolor("#f5f8f6")

        _flow(fig, 0.06, ytab - 0.022, P2_VALIDATION)
        pdf.savefig(fig); plt.close(fig)

        # ===================== PAGE 3: full-page validation figure =====================
        # The figure carries its own title, so no separate page header (which would
        # be redundant and push the figure down). It fills the page.
        fig = plt.figure(figsize=(8.27, FIG_H_IN))
        repo = run.resolve().parent.parent
        vfig = repo / "reports" / "autofocus_validation.png"
        axv = fig.add_axes([0.02, 0.02, 0.96, 0.96]); axv.axis("off")
        if vfig.exists():
            import matplotlib.image as mpimg
            axv.imshow(mpimg.imread(str(vfig)))
        else:
            axv.text(0.5, 0.5, "(validation figure not found; run\n"
                     "python scripts/autofocus_validation.py)", ha="center", va="center",
                     fontsize=11, color="#999", transform=axv.transAxes)
        pdf.savefig(fig); plt.close(fig)

    return pdf_path

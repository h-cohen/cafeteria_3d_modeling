"""Systematic-error estimate for the unmodelled physics: multiple Coulomb
scattering (MCS) in the ceiling.

The forward model treats opacity as pure absorption along straight rays. A muon
that crosses a concrete beam is not only sometimes stopped -- it is always
deflected, by the Highland RMS angle

    theta0 = (13.6 MeV / (beta c p)) * sqrt(L/X0) * (1 + 0.038 ln(L/X0)).

The detector measures the POST-scatter direction, so the ceiling image is
blurred laterally by ~theta0 * lever. This script quantifies that blur for the
campaign geometry, flux-weights it over an approximate sea-level muon spectrum,
and translates it into the observable biases (beam widening, amplitude
suppression; positions are unbiased by symmetry). Numbers are order-estimates:
the spectrum shape is approximate (~30%), which is ample for a bound.

    python scripts/mcs_systematics.py [--run runs/production]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

M_MU = 0.10566  # GeV
X0_CONCRETE_M = 0.1155  # radiation length of standard concrete, metres
BEAM_DEPTH_M = 0.30  # structural beam crossing (normal incidence)


def theta0(p_gev: np.ndarray, L_m: float) -> np.ndarray:
    """Highland RMS projected scattering angle (radians)."""
    x = L_m / X0_CONCRETE_M
    beta = p_gev / np.sqrt(p_gev**2 + M_MU**2)
    return 13.6e-3 / (beta * p_gev) * np.sqrt(x) * (1 + 0.038 * np.log(x))


def spectrum_weight(p_gev: np.ndarray) -> np.ndarray:
    """Approximate sea-level muon momentum spectrum shape: flat below a few GeV,
    ~p^-2.7 above (mean energy ~4 GeV). Shape-only; adequate for weighting."""
    return 1.0 / (1.0 + (p_gev / 3.5) ** 2.7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/production")
    ap.add_argument("--lever", type=float, default=7.0, help="ceiling height / lever arm (m)")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    t_max = cfg["binning"]["t_max"]
    rebin = cfg["binning"]["rebin"]
    bin_w = 2 * t_max / (800 * t_max / 2 / rebin)  # raw 800 bins over +/-2 tan-units
    fwhm = None
    f = run / "enhance" / "dipclean_metrics.json"
    if f.exists():
        fwhm = json.loads(f.read_text()).get("beam_fwhm_m")
    fwhm = fwhm or 0.53

    z = args.lever
    ps = np.geomspace(0.2, 100, 4000)
    w = spectrum_weight(ps)
    th = theta0(ps, BEAM_DEPTH_M)

    out = {
        "geometry": {"lever_m": z, "bin_width_tan": round(bin_w, 4),
                     "bin_footprint_m": round(bin_w * z, 3),
                     "beam_fwhm_measured_m": fwhm,
                     "beam_depth_m": BEAM_DEPTH_M,
                     "L_over_X0": round(BEAM_DEPTH_M / X0_CONCRETE_M, 2)},
        "theta0_mrad": {f"{p:g}GeV": round(1e3 * float(theta0(np.array([p]), BEAM_DEPTH_M)[0]), 2)
                        for p in (1.0, 3.0, 10.0)},
        "blur_at_ceiling_m": {},
        "flux_weighted": {},
    }
    for p_min in (0.3, 0.5, 1.0):
        sel = ps >= p_min
        th_mean = float(np.trapezoid(th[sel] * w[sel], ps[sel]) / np.trapezoid(w[sel], ps[sel]))
        sigma = th_mean * z
        widened = float(np.hypot(fwhm, 2.355 * sigma))
        out["flux_weighted"][f"pmin_{p_min}GeV"] = {
            "theta0_mrad": round(1e3 * th_mean, 2),
            "blur_sigma_m": round(sigma, 3),
            "fwhm_widening_frac": round(widened / fwhm - 1, 3),
            "amplitude_suppression_frac": round(1 - fwhm / widened, 3),
        }
    for p in (1.0, 3.0, 10.0):
        out["blur_at_ceiling_m"][f"{p:g}GeV"] = round(
            float(theta0(np.array([p]), BEAM_DEPTH_M)[0]) * z, 3)

    dest = Path("reports/mcs_systematics.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")

    fw = out["flux_weighted"]["pmin_0.5GeV"]
    print(f"theta0: {out['theta0_mrad']} mrad  (L/X0 = {out['geometry']['L_over_X0']})")
    print(f"blur at ceiling: {out['blur_at_ceiling_m']} m")
    print(f"flux-weighted (p>0.5 GeV): theta0 = {fw['theta0_mrad']} mrad -> "
          f"sigma = {fw['blur_sigma_m']} m at the ceiling")
    print(f"vs angular-bin footprint {out['geometry']['bin_footprint_m']} m "
          f"and measured beam FWHM {fwhm} m")
    print(f"-> beam FWHM widening {100*fw['fwhm_widening_frac']:.1f}% ; "
          f"peak amplitude suppression {100*fw['amplitude_suppression_frac']:.1f}% ; "
          f"positions unbiased (symmetric)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

"""Does single-detector refocusing help? Evidence for/against, from the real campaign.

The DAQ writes single-detector REFOCUSED images: XY01m/02m/05m/07m/10m are the raw
txty map sheared track-by-track by

    t_corr = t + b/H          (b = bottom-layer hit position, H = assumed height)

which collapses every track from a source at height H onto a common vertex. This is
the intra-detector analogue of two-detector parallax focusing, at the scale of the
detector aperture instead of the stereo baseline. This script establishes three things
and writes reports/refocus_analysis.{png,json}:

1. The maps really are that shear, not a metric rescaling (as the code once assumed).
   Under the shear, E[t_corr]*H - E[t_inf]*H = E[b], independent of H.
2. Two ways to score focus, one scene-specific and one scene-agnostic. Naive sharpness
   (gradient energy, Laplacian variance, total variation, raw high-frequency power) FAILS:
   these maps are noise-dominated at high frequency, and the shear smooths noise along
   with signal, so raw sharpness just tracks how much shear was applied. Beam modulation
   depth (Fourier amplitude at the KNOWN beam pitch) works, but needs that pitch in
   advance. Signal-band power -- integrate noise-corrected spectral power over the band
   the scene itself rises above the Poisson floor, detected from the data -- needs no
   prior knowledge of the scene and separates focused from defocused better still.
3. Both are REAL but WEAK: they peak at the independently-known ceiling height in each
   detector separately, but are nearly flat above ~5 m -- so they exclude low ceilings
   decisively and cannot localize a high one. The blur removed is small regardless,
   because the accepted spread of b within one angular bin is far narrower than the
   detector.

    python scripts/refocus_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from muontomo import beams as B  # noqa: E402
from muontomo.focus import _windowed_spectrum, signal_band, spectral_focus  # noqa: E402
from muontomo.calibration import compute_transmission, prepare_angular_hist
from muontomo.config import BinningConfig, RunConfig
from muontomo.io import load_dataset, load_root_hist2d

HISTS = [("XY01m", 1.0), ("XY02m", 2.0), ("XY05m", 5.0), ("XY07m", 7.0), ("XY10m", 10.0)]
BAR_ORIGIN_M = 0.1912  # detector centre in raw bar coordinates (see shear_signature)
BEAM_PITCH_M = 1.58
CEILING_M = 7.0


def shear_signature(ds) -> dict:
    """E[b] recovered from each refocus height. Constant => the maps are the shear."""
    out = {}
    for src in ("pos0", "pos1", "sky"):
        raw = load_root_hist2d(ds.sources[src], "txty")
        m0 = [(raw.values.sum(1) * raw.xcenters).sum() / raw.values.sum(),
              (raw.values.sum(0) * raw.ycenters).sum() / raw.values.sum()]
        rows = {}
        for name, h in HISTS:
            v = load_root_hist2d(ds.sources[src], name)
            mx = (v.values.sum(1) * v.xcenters).sum() / v.values.sum()
            my = (v.values.sum(0) * v.ycenters).sum() / v.values.sum()
            rows[name] = {"E_x_bot_m": round((mx - m0[0]) * h, 5),
                          "E_y_bot_m": round((my - m0[1]) * h, 5)}
        vals = [r["E_x_bot_m"] for r in rows.values()]
        out[src] = {"per_height": rows, "spread_m": round(max(vals) - min(vals), 5),
                    "mean_m": round(float(np.mean(vals)), 5)}
    return out


def _tmap(ds, cfg, pos: str, hist: str, rebin: int = 2):
    b = BinningConfig(hist=hist, t_max=1.0, rebin=rebin,
                      refocus_origin_m=None if hist == "txty" else BAR_ORIGIN_M)
    cafe = prepare_angular_hist(load_root_hist2d(ds.sources[pos], hist), b)
    sky = prepare_angular_hist(load_root_hist2d(ds.sources["sky"], hist), b)
    return compute_transmission(cafe, sky, cfg.calibration, pose_id=pos)


def modulation(ds, cfg, pos: str, hist: str, t_window: float = 0.5) -> float:
    """Beam modulation depth for one (position, refocus height) -- see _modulation_tm."""
    return _modulation_tm(_tmap(ds, cfg, pos, hist), t_window)


def _modulation_tm(tm, t_window: float = 0.5) -> float:
    """Beam modulation depth at the KNOWN beam period, in a fixed angular window.

    The scene-specific metric: it needs no peak detection, and holding the window and the
    period fixed in tan-space keeps out the lever-arm bias that disqualifies
    gradient-energy sharpness (see report Sec. 3.2). But it can only be computed at all
    because the beam pitch is known in advance.
    """
    t, p = B.profile(tm, "x")
    sel = np.isfinite(p) & (np.abs(t) < t_window)
    tt, pp = t[sel], p[sel]
    if len(tt) < 16:
        return float("nan")
    pp = pp - np.polyval(np.polyfit(tt, pp, 3), tt)
    freq = np.fft.rfftfreq(len(tt), tt[1] - tt[0])
    amp = np.abs(np.fft.rfft(pp * np.hanning(len(pp))))
    i = int(np.argmin(np.abs(freq - CEILING_M / BEAM_PITCH_M)))
    return float(amp[max(i - 1, 0): i + 2].max() / len(pp))


def radial_power_profile(tm, t_window: float = 0.5, n_bands: int = 60):
    """Radially-averaged power spectrum and its Poisson floor, in cycles per tan-unit.

    The diagnostic behind `signal_band`. Power and floor are returned separately, rather
    than as a ratio, so that a plot can show the measured spectrum crossing the noise
    level -- which is what "the scene has detectable structure here" actually means.
    Returns (freq_tan, power, noise_floor, bin_width_tan).
    """
    power, noise, r = _windowed_spectrum(tm, t_window)
    if power is None:
        return None
    bin_t = float(tm.txedges[1] - tm.txedges[0])
    r_tan = r / bin_t  # cycles/bin (index space) -> cycles/tan-unit
    edges = np.linspace(0, float(r_tan.max()), n_bands + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([
        power[(r_tan >= edges[i]) & (r_tan < edges[i + 1])].mean()
        if ((r_tan >= edges[i]) & (r_tan < edges[i + 1])).any() else np.nan
        for i in range(len(ctr))
    ])
    return ctr, prof, noise, bin_t


def beam_widths(ds, cfg, hist: str) -> dict:
    """Two-detector world-projected beam FWHM at the ceiling -- the practical payoff."""
    tmaps = {p: _tmap(ds, cfg, p, hist, rebin=8) for p in ("pos0", "pos1")}
    grid = np.linspace(-4, 6, 600)
    profs = {p: B.profile(t, "x") for p, t in tmaps.items()}
    stack = [B._world_profile(*profs[p], cfg.geometry.pose(p), "x", CEILING_M, grid)
             for p in profs]
    prof = B._nanmean(np.stack(stack), axis=0)
    peaks = B.beam_peaks_subbin(grid, prof)
    base = np.nanpercentile(prof, 20)
    fw = []
    for c in peaks:
        i = int(np.argmin(np.abs(grid - c)))
        half = (prof[i] + base) / 2
        lo = hi = i
        while lo > 0 and prof[lo] > half:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi] > half:
            hi += 1
        fw.append(grid[hi] - grid[lo])
    return {"n_beams": len(peaks), "fwhm_m": [round(float(v), 3) for v in fw],
            "mean_fwhm_m": round(float(np.mean(fw)), 4) if fw else None,
            "contrast": round(float(np.nanstd(prof) / np.nanmean(prof)), 4)}


def bootstrap_metrics(ds, cfg, band, n_boot: int, seed: int = 0) -> dict:
    """Poisson-bootstrap both focus metrics over the refocus stack.

    Same resampling as muontomo.uncertainty, so the error bars mean what they mean
    everywhere else in the project: counting statistics propagated through the identical
    computation. Also records WHICH height each replica peaks at -- with only five canned
    heights and a curve that saturates rather than peaks sharply, the argmax distribution
    is a far more honest statement of "where it focuses" than a single number.
    """
    from muontomo.uncertainty import resample_tmaps

    base = {(h, p): _tmap(ds, cfg, p, h) for h, _ in HISTS for p in ("pos0", "pos1")}
    rng = np.random.default_rng(seed)
    spec = np.zeros((n_boot, len(HISTS)))
    modu = np.zeros((n_boot, len(HISTS)))
    for i in range(n_boot):
        for k, (hname, _) in enumerate(HISTS):
            rep = resample_tmaps({p: base[(hname, p)] for p in ("pos0", "pos1")}, rng)
            spec[i, k] = np.mean([spectral_focus(rep[p], band) for p in rep])
            modu[i, k] = np.mean([_modulation_tm(rep[p]) for p in rep])

    def summarize(a):
        mean = a.mean(axis=0)
        norm = float(mean.max())
        am = a.argmax(axis=1)
        return {
            "mean": [round(float(v / norm), 4) for v in mean],
            "sd": [round(float(v / norm), 4) for v in a.std(axis=0, ddof=1)],
            "peak_fraction": [round(float((am == i).mean()), 4) for i in range(len(HISTS))],
        }

    return {"n_boot": n_boot, "heights_m": [h for _, h in HISTS],
            "spectral": summarize(spec), "modulation": summarize(modu)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/production.json")
    ap.add_argument("--out", default="reports/refocus_analysis")
    ap.add_argument("--n-boot", type=int, default=400,
                    help="Poisson bootstrap replicas for the focus-metric error bands")
    args = ap.parse_args()

    cfg = RunConfig.load(args.config)
    ds = load_dataset(cfg.data)

    res = {"shear_signature": shear_signature(ds), "modulation": {},
           "spectral": {}, "beam_widths": {}}
    # The signal band is detected once, from the un-refocused map, and reused for every H:
    # the scene's angular structure does not depend on the height we assume.
    tm0 = _tmap(ds, cfg, "pos0", "txty")
    band = signal_band(tm0)
    res["signal_band_cycles_per_bin"] = [round(b, 4) for b in band] if band else None
    prof = radial_power_profile(tm0)
    if prof:
        freq_tan, power, noise, bin_t = prof
        res["radial_power"] = {
            "freq_tan": [round(f, 4) for f in freq_tan],
            "power": [None if not np.isfinite(v) else round(float(v), 4) for v in power],
            "noise_floor": round(float(noise), 4),
            "bin_width_tan": round(bin_t, 5),
            "band_tan": [round(b / bin_t, 4) for b in band] if band else None,
            "pitch_freq_tan": round(CEILING_M / BEAM_PITCH_M, 4),
            "ceiling_m": CEILING_M,
        }
    for name, h in HISTS + [("txty", float("inf"))]:
        res["modulation"][name] = {
            "height_m": h,
            "pos0": round(modulation(ds, cfg, "pos0", name), 6),
            "pos1": round(modulation(ds, cfg, "pos1", name), 6),
        }
        if band:
            res["spectral"][name] = {
                "height_m": h,
                "pos0": round(spectral_focus(_tmap(ds, cfg, "pos0", name), band), 2),
                "pos1": round(spectral_focus(_tmap(ds, cfg, "pos1", name), band), 2),
            }
    for name in ("txty", "XY07m"):
        res["beam_widths"][name] = beam_widths(ds, cfg, name)

    mod = res["modulation"]
    finite = {k: v for k, v in mod.items() if np.isfinite(v["height_m"])}
    best = max(finite, key=lambda k: 0.5 * (finite[k]["pos0"] + finite[k]["pos1"]))
    hi = [v for k, v in finite.items() if v["height_m"] >= 5.0]
    means = [0.5 * (v["pos0"] + v["pos1"]) for v in hi]
    sp = res["spectral"]
    spf = {k: v for k, v in sp.items() if np.isfinite(v["height_m"])}
    sp_best = max(spf, key=lambda k: 0.5 * (spf[k]["pos0"] + spf[k]["pos1"])) if spf else None
    sp_hi = [0.5 * (v["pos0"] + v["pos1"]) for v in spf.values() if v["height_m"] >= 5.0]
    res["verdict"] = {
        "best_refocus_height_m": finite[best]["height_m"],
        "spectral_best_height_m": spf[sp_best]["height_m"] if sp_best else None,
        "spectral_flatness_above_5m_frac": (
            round((max(sp_hi) - min(sp_hi)) / max(sp_hi), 4) if sp_hi else None),
        "spectral_gain_over_unrefocused_frac": (
            round(0.5 * (sp["XY07m"]["pos0"] + sp["XY07m"]["pos1"])
                  / (0.5 * (sp["txty"]["pos0"] + sp["txty"]["pos1"])) - 1, 4)
            if "XY07m" in sp and "txty" in sp else None),
        "flatness_above_5m_frac": round((max(means) - min(means)) / max(means), 4),
        "fwhm_gain_frac": round(
            1 - res["beam_widths"]["XY07m"]["mean_fwhm_m"]
            / res["beam_widths"]["txty"]["mean_fwhm_m"], 4),
    }

    if band and args.n_boot > 0:
        print(f"bootstrapping focus metrics ({args.n_boot} replicas)...")
        res["bootstrap"] = bootstrap_metrics(ds, cfg, band, args.n_boot)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    _plot(res, dest.with_suffix(".png"))
    _plot_metrics_figure(res, dest.with_name("refocus_focus_curve.png"))
    _plot_spectrum(res, dest.with_name("refocus_signal_band.png"))

    v = res["verdict"]
    print(f"shear E[b] spread across H: "
          f"{res['shear_signature']['pos0']['spread_m']:.5f} m "
          f"(mean {res['shear_signature']['pos0']['mean_m']:.4f}) -> the maps ARE the shear")
    print(f"modulation peaks at H = {v['best_refocus_height_m']} m, "
          f"but varies only {100*v['flatness_above_5m_frac']:.1f}% over 5-10 m")
    print(f"beam FWHM {res['beam_widths']['txty']['mean_fwhm_m']:.3f} m -> "
          f"{res['beam_widths']['XY07m']['mean_fwhm_m']:.3f} m "
          f"({100*v['fwhm_gain_frac']:.1f}% narrower)")
    if v.get("spectral_best_height_m"):
        print(f"scene-agnostic spectral focus peaks at H = {v['spectral_best_height_m']} m "
              f"(band {res['signal_band_cycles_per_bin']} cyc/bin), "
              f"{100*v['spectral_flatness_above_5m_frac']:.1f}% over 5-10 m, "
              f"{100*v['spectral_gain_over_unrefocused_frac']:.0f}% above un-refocused")
    print(f"wrote {dest.with_suffix('.json')} and {dest.with_suffix('.png')}")


def _plot_metrics_figure(res: dict, path: Path) -> None:
    """Scene-specific vs scene-agnostic focus, with bootstrap uncertainty. Paper Fig. 3.

    (a) Both metrics against assumed refocus height, with +/-1 and +/-2 sigma bands from
        the Poisson bootstrap.
    (b) The argmax distribution: with five coarse heights and a curve that saturates
        rather than peaks, "where it focuses" is a probability, not a number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = res.get("bootstrap")
    if not bs:
        return
    hs = np.array(bs["heights_m"], float)
    n = bs["n_boot"]
    blue, orange = "#2a78d6", "#eb6834"

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(11.4, 3.7),
        gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)

    ax.axvline(CEILING_M, color="#111", ls=":", lw=1.5, zorder=1,
               label=f"true ceiling {CEILING_M:g} m")
    series = (
        (bs["spectral"], blue, "signal-band power (scene-agnostic)", "-", "o"),
        (bs["modulation"], orange, "beam modulation (scene-specific)", "--", "s"),
    )
    for blk, c, lab, ls, mk in series:
        m = np.array(blk["mean"], float)
        sd = np.array(blk["sd"], float)
        ax.fill_between(hs, m - 2 * sd, m + 2 * sd, color=c, alpha=0.12, lw=0, zorder=2)
        ax.fill_between(hs, m - sd, m + sd, color=c, alpha=0.28, lw=0, zorder=3)
        ax.plot(hs, m, ls, marker=mk, lw=2.0, ms=6.5, color=c, zorder=4, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:g}" for h in hs])
    ax.set_xlabel("assumed refocus height $H$ (m)", fontsize=11)
    ax.set_ylabel("focus metric (normalised)", fontsize=11)
    ax.set_title(r"(a) focus curves with $\pm1\sigma$ and $\pm2\sigma$ bands", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

    w = 0.38
    x = np.arange(len(hs))
    for off, blk, c, lab in ((-w / 2, bs["spectral"], blue, "signal-band power"),
                             (w / 2, bs["modulation"], orange, "beam modulation")):
        f = np.array(blk["peak_fraction"], float)
        err = 100 * np.sqrt(f * (1 - f) / n)          # binomial error on the fraction
        axb.bar(x + off, 100 * f, w, color=c, label=lab, yerr=err, capsize=3,
                error_kw=dict(lw=1.1, ecolor="#333"))
        for i in range(len(hs)):
            if 100 * f[i] > 2:
                axb.text(x[i] + off, 100 * f[i] + err[i] + 1.6, f"{100*f[i]:.0f}",
                         ha="center", fontsize=8.5)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{h:g}" for h in hs])
    axb.set_xlabel("assumed refocus height $H$ (m)", fontsize=11)
    axb.set_ylabel("replicas peaking here (%)", fontsize=11)
    axb.set_title(f"(b) where the peak lands ({n} replicas)", fontsize=11)
    axb.set_ylim(0, 82)
    axb.grid(alpha=0.3, axis="y")
    axb.legend(fontsize=8.5, loc="upper left")
    axb.tick_params(labelsize=10)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_spectrum(res: dict, path: Path) -> None:
    """How signal_band picks its band. Explanatory, for the memo -- not the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rp = res.get("radial_power")
    if not rp:
        return
    f = np.array(rp["freq_tan"], float)
    pw = np.array([np.nan if v is None else v for v in rp["power"]], float)
    noise, z, band = rp["noise_floor"], rp["ceiling_m"], rp.get("band_tan")
    size = z / np.maximum(f, 1e-9)          # structure size at the ceiling, metres
    keep = (size > 0.28) & (size < 30) & np.isfinite(pw)

    fig, ax = plt.subplots(figsize=(6.6, 3.3), constrained_layout=True)
    if band:
        ax.axvspan(z / band[1], z / band[0], color="#2a78d6", alpha=0.13, zorder=0,
                   label="detected signal band\n(found from the data alone)")
    ax.plot(size[keep], pw[keep], "-", lw=1.9, color="#111", zorder=3,
            label="measured power")
    ax.axhline(noise, color="#c0392b", ls="--", lw=1.6, zorder=2,
               label="Poisson noise floor")
    ax.axvline(BEAM_PITCH_M, color="#eb6834", ls="-", lw=2.2, zorder=4,
               label=f"beam pitch {BEAM_PITCH_M:g} m\n(what the specific metric needs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 20)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_ylim(top=float(np.nanmax(pw[keep])) * 3.0)
    ax.set_xlabel("structure size at the ceiling (m)", fontsize=10.5)
    ax.set_ylabel("spectral power (arb.)", fontsize=10.5)
    ax.set_title("which structure sizes rise above the noise", fontsize=11)
    ax.legend(fontsize=7.6, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=9.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    ax = axes[0]
    sig = res["shear_signature"]
    for src in sig:
        hs = [HISTS[i][1] for i in range(len(HISTS))]
        vs = [sig[src]["per_height"][n]["E_x_bot_m"] for n, _ in HISTS]
        ax.plot(hs, vs, "o-", label=f"{src} (spread {1e3*sig[src]['spread_m']:.2f} mm)")
    ax.set(xscale="log", xlabel="assumed refocus height H (m)",
           ylabel=r"recovered $E[b]$ (m)", ylim=(0.18, 0.20),
           title="1. The maps are the shear\n"
                 r"$E[t_{corr}]H-E[t_\infty]H=E[b]$, constant in $H$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    mod = res["modulation"]
    fin = [(v["height_m"], v["pos0"], v["pos1"]) for v in mod.values()
           if np.isfinite(v["height_m"])]
    fin.sort()
    hs = [f[0] for f in fin]
    ax.plot(hs, [f[1] for f in fin], "o-", label="pos0")
    ax.plot(hs, [f[2] for f in fin], "s-", label="pos1")
    inf = mod["txty"]
    ax.axhline(0.5 * (inf["pos0"] + inf["pos1"]), color="gray", ls="--",
               label=r"un-refocused (txty, $H=\infty$)")
    ax.axvline(CEILING_M, color="k", ls=":", label=f"true ceiling {CEILING_M} m")
    ax.set(xscale="log", xlabel="assumed refocus height H (m)",
           ylabel="beam modulation depth",
           title="2. Focusing is real but weak\n"
                 f"peaks at the true height; only "
                 f"{100*res['verdict']['flatness_above_5m_frac']:.0f}% of range over 5-10 m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    bw = res["beam_widths"]
    names = ["txty", "XY07m"]
    ax.bar(names, [bw[n]["mean_fwhm_m"] for n in names],
           color=["#888", "#2a7"], width=0.55)
    for i, n in enumerate(names):
        ax.text(i, bw[n]["mean_fwhm_m"] + 0.008, f"{bw[n]['mean_fwhm_m']:.3f} m",
                ha="center")
    ax.set(ylabel="mean beam FWHM at the ceiling (m)", ylim=(0, 0.62),
           title="3. The practical gain is small\n"
                 f"{100*res['verdict']['fwhm_gain_frac']:.0f}% narrower: the accepted "
                 r"spread of $b$" "\nper angular bin is far below the aperture")
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

# Re: single-detector focusing — how to score it

Quick recap of where we left off: the DAQ's `XY01m..XY10m` histograms are exactly your
`t_corr = t + b/H`, at H = 1, 2, 5, 7, 10 m — confirmed by `E[t_corr]·H − E[t_∞]·H = E[b]`
holding constant to 0.25 mm across a decade in H. Our code had wrongly called that a pure
rescaling with no depth information; fixed. This note is about the question that follows
from that: **how do you actually score focus?**

## The obvious metric doesn't generalise

Beam modulation depth — the Fourier amplitude of the opacity profile at the *known* beam
period — does work here: it prefers 7.0 m, our independently-known ceiling, in each
detector separately. But it only works at all because we already know the pitch. On an
unrecognised scene there's nothing to look for at a known frequency, so it isn't a general
answer to your proposal — it's an answer that happens to work on a periodic ceiling.

I tried the obvious generic replacements — gradient energy, Laplacian variance, total
variation, raw high-frequency power — and **all four fail**, informatively. At these
exposures the maps are noise-dominated at high spatial frequency (Poisson floor ~3× the
signal power there), and the refocusing shear smooths noise along with signal. So any raw
sharpness measure just tracks *how much shear was applied*, monotonic in 1/H — every one of
the four peaks at the edge of the scan, not at the true height.

## A metric that needs no prior knowledge of the scene

The fix is to separate signal from noise by their statistics, not their shape. Per-bin
opacity variance is known analytically from the counts (Poisson, already propagated), so the
expected noise floor of the power spectrum is computable, not fitted. Subtract it from the
2-D power spectrum; what survives is signal.

1. **Detect the band.** Bin the spectrum radially, flag any band where the noise-subtracted
   power still exceeds the floor. Nothing about the scene is assumed — no pitch, no shape —
   only the data and its own noise model.
2. **Score focus** as the total noise-corrected power inside that detected band.

Two results (`focus.signal_band`, `focus.spectral_focus`):

- **The detected band contains your known pitch without being told about it.** The attached
  spectrum figure shows it: power rises above the Poisson floor for structures between
  ~0.75 and ~11 m, and the 1.58 m beam pitch sits inside that range. The general method
  rediscovers what the specific one had to be handed.
- **It's the better metric here.** It separates the refocused map from the un-refocused one
  by 16%, against 1% for modulation depth, and it localises the peak more sharply — see
  below.

So for an unrecognised scene, this is what I'd reach for: no prior structure needed, better
discrimination, and it degrades gracefully toward modulation depth if a scene happens to be
strongly periodic (the detected band just narrows around that one frequency).

## How well does either actually locate the height?

I put proper error bars on this rather than eyeballing the curves — a 400-replica Poisson
bootstrap of the raw counts, the same resampling we use for every other uncertainty in the
pipeline. The answer is weaker than the curves suggest:

| peaks at | 5 m | 7 m | 10 m |
|---|---|---|---|
| signal-band power | 1% | **66%** | 32% |
| beam modulation | 12% | **53%** | 35% |

Both prefer the true 7 m ceiling, and the scene-agnostic metric does so more decisively
(66% vs 53%) — another point in its favour. But two-thirds is a preference, not a
measurement: across 5–10 m the metrics vary by only ~7%, comparable to their own ±1σ bands.
What they *do* say emphatically is that the ceiling isn't at 1 or 2 m — those are excluded
by many σ.

That flatness is geometric, not a metric failing: residual blur scales as
`A·|1/H − 1/H_true|`, which barely moves once both heights are large. No focus measure
recovers sensitivity there; only baseline does. It also doesn't
touch the earlier finding that refocused maps must stay out of the inversion, or that the
assume-H → sharpen → re-measure-H loop has no leverage on this geometry (~7% of contrast
isn't enough to iterate on, whichever metric drives it) — for the outer loop the
two-detector cross-validation score remains the right scorer.

Reproduce with `python scripts/refocus_analysis.py` → `reports/refocus_analysis.{png,json}`,
`reports/refocus_focus_curve.png` (the bootstrap comparison) and
`reports/refocus_signal_band.png` (the spectrum above).

---

**Still open from last time, for the record:** your `d = 2.4 m` versus our calibrated
1.916 m matters because `z ∝ d` — 6.96 m of ceiling versus 8.71 m. Each detector measures
the beam angular period on its own (0.2325 tan-units both, no baseline needed), so
`pitch = z × period` closes the (baseline, height, pitch) triple once one length is
surveyed. Where does 2.4 m come from — is it a tape measure, or a prior fit? And the
triangulation least-squares you proposed is built and gives an independent
z = 6.955 ± 0.086 m, now in every run's autofocus report.

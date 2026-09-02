# Muon Tomography: 3D Room Reconstruction

Reconstructs a 3D density-length volume of a cafeteria room's ceiling from
cosmic-ray muon flux measured at two detector positions, calibrated against a
clear-sky reference. The ceiling height (7.0 m) is measured from the data
itself by a validated parallax autofocus. See `detector_summary.md` for the
detector itself, and `reports/voxel_reconstruction_report.pdf` for a
paper-style write-up of the full-room voxel model and the two-detector limits.

## Data

`data/` holds three ROOT files of pre-made histograms (no event trees):

- `HistsOutSkyRoofRuns37-77.root` — clear-sky reference (28.5M tracks)
- `HistsOutDataCafePos0.root`, `HistsOutDataCafePos1.root` — detector inside
  the cafeteria at two positions ~1.9 m apart, horizontal movement only,
  same floor (6.3M / 2.4M tracks)

The angular flux map `txty` (tan θx vs tan θy, 800×800 bins over ±2) is the
primary analysis histogram. `XY01m..XY10m` are **single-detector refocused**
maps — `txty` sheared track-by-track by `t_corr = t + b/H` at H = 1/2/5/7/10 m.
They are genuinely not a rescaling of `txty` and do carry (weak) depth
information; see "Single-detector refocusing" below for what they are worth.

## Install

```
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
playwright install chromium              # for the viewer smoke test
```

Optional (only the `dip` / `pnp` / `dipclean` enhancers need them — everything
else, including the test suite, runs without):

```
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install scikit-image pywavelets
```

## Physics model

For each angular bin at each position: transmission `T = N_cafe / (s·N_sky)`,
opacity `λ = -ln T ≈ ∫κ dl` along that bin's ray. The detector aperture
(`aperture_m = 0.65`, but see the open item under "Geometry self-calibration" —
the measured active width is 0.354 m) is comparable to the beam pitch projected
to the detector, so each
bin is modeled as a bundle of parallel sub-rays across the aperture
(angle-dependent, narrowing with the 4-layer coincidence requirement), not a
pinhole. With only two views this is severely limited-angle tomography: expect
vertical elongation; TV regularization and a layered (height-scan) model
counter it. See `muontomo/reconstruct.py` and `muontomo/raycast.py`, and the
depth-degeneracy quantification in `reports/voxel_reconstruction_report.pdf`.

## Pipeline

```
scripts/inspect_data.py        # Stage 0: dump histogram metadata, convention checks
scripts/refocus_analysis.py     # single-detector refocusing: is the DAQ's XY0*m shear useful?
muontomo.selfcal                # geometry self-calibration (Stage A diagnostic, Stage B authoritative)
muontomo.reconstruct            # produce a run: volume.npz + holdout volumes
muontomo.evaluate                # metrics.json scorecard + PNG set + autofocus report
muontomo.focus                   # ceiling-height autofocus: CV height-scan + quick-look + height map
muontomo.backproject             # model-free backprojection of -ln(T) onto the ceiling plane
muontomo.beams                   # model-free beam verification (joint LSQ triangulation + positions)
muontomo.enhance                 # enhancement suite: guided | dip | pnp | clean | dipclean
muontomo.uncertainty             # bootstrap error bars: layer sigma-map, beam pos/amp errors, z* CI
muontomo.compare                 # IMPROVED/REGRESSED verdict between two runs
muontomo.phantom                 # synthetic ground truth for rigorous algorithm gating
muontomo.viewer.build            # self-contained viewer.html
muontomo.viewer.smoke            # Playwright check of the viewer
```

Configs: `configs/production.json` (layered solve at the autofocused 7.0 m —
the quantitative reference product) and `configs/full3d.json` (unconstrained
full-room solve over 1.9M voxels — the qualitative room-scale view; drop its
`volume.npz` into a run as `volume_full3d.npz` to enable the viewer's
full-room voxel toggle).

### The agent feedback loop

```
1. edit algorithm/config
2. python -m muontomo.reconstruct --config configs/production.json --out runs/expNN
3. python -m muontomo.evaluate --run runs/expNN
4. Read runs/expNN/metrics.json + runs/expNN/images/panel_summary.png
     high deviance/chi2       -> read residual_*.png
     large chi2_aligned_gap   -> fix geometry, not the algorithm
     low/negative cv_pearson  -> overfitting; strengthen regularization
     large z_eff_width_m      -> limited-angle smearing along rays
     autofocus_z_m != solve z -> re-solve at the autofocused height
     beam_offset_m > 0.15     -> reconstruction moved the beams; distrust it
5. python -m muontomo.compare runs/prev runs/expNN
6. algorithm changes additionally gated on phantoms (real ground truth):
     python -m muontomo.phantom --preset beams --seed 42 --out phantoms/p1
     python -m muontomo.reconstruct --config <cfg pointed at phantoms/p1> --out runs/ph_expNN
     python -m muontomo.evaluate --run runs/ph_expNN --truth phantoms/p1/truth_volume.npz
7. converged -> python -m muontomo.enhance --run runs/expNN --method all
              -> python -m muontomo.viewer.build --run runs/expNN
              -> python -m muontomo.viewer.smoke --run runs/expNN
```

**Metric roles**: phantom-truth metrics (RMSE/SSIM/IoU/z-error) gate algorithm
changes — they're the only metrics with real ground truth. Fidelity and
cross-position-CV metrics gate calibration/geometry changes. The model-free
beam verification (`muontomo.beams`) gates every displayed product: peak
positions must match the raw data (mean |offset| ≤ 0.15 m, five beams).
Structural metrics (periodicity, contrast) are advisory only.

Every `runs/<name>/` is reproducible: `config.json` snapshots the full config
+ git hash; `evaluate` only ever adds `metrics.json`, `images/` and
`autofocus_report.pdf`, never touches the volume.

## Ceiling-height autofocus

The layered inversion needs the ceiling height as input; `muontomo.focus`
measures it from the data by a plane-sweep autofocus:

- **Primary**: a cross-validation height-scan — fit a thin layer to one
  detector, score how well it predicts the other (trimmed residual, two-stage
  coarse→fine sweep). Alias-robust; selects **7.0 m** on this campaign.
- **Quick-look**: model-free two-view NCC + a windowed height map (planarity
  check). Fast, shown in the viewer HUD, biased by non-beam structure.
- **Validated on synthetic ground truth**: campaign-geometry phantoms with
  ceilings injected at 6.6/7.0/7.4 m are recovered to ~0.2 m
  (`python scripts/autofocus_validation.py` →
  `reports/autofocus_validation.png`).
- **Independent cross-check**: `muontomo.beams` triangulates the height by a
  joint least-squares ray intersection over every matched beam in both axes
  (`beams.triangulate`), giving **6.955 ± 0.086 m** — model-free, and
  independent of both the solver and the CV scan. Sub-bin peak refinement
  (`beam_peaks_subbin`) is essential: with a 1.78 m baseline, one 0.04 tan bin
  of quantization moves a single-pair closed-form height by ~1 m.
  Caveat: `z = dx/Δt` scales with the assumed baseline, so a 1% baseline error
  is worth 0.07 m — more than the statistical error bar.
- **Scale closure** (`beams.scale_closure`): angles alone fix only the ratio
  `z/d`. Each detector separately measures the beam pattern's angular period
  (0.2329 / 0.2321 tan-units — no baseline, pose or height involved), and
  `pitch = z × period`, so baseline, height and pitch form a closed triple:

  | surveyed baseline | ceiling height | implied beam pitch |
  | --- | --- | --- |
  | 1.92 m (this calibration) | 6.96 m | 1.62 m |
  | 2.40 m | 8.71 m | 2.03 m |

  **One external length is required** and no algorithm work substitutes for it:
  a tape measure on either the detector separation or the ceiling beam spacing
  fixes all three. The run's `scale_closure` block reports `z_per_baseline` and
  `pitch_per_baseline` so any surveyed `d` can be substituted without re-running.

Every `evaluate` regenerates `runs/<run>/autofocus_report.pdf` — a 3-page
plain-language report of the method, this run's numbers, and the validation.

## Single-detector refocusing (evaluated, not adopted)

The DAQ's `XY01m/02m/05m/07m/10m` histograms are the raw `txty` map sheared
track-by-track by `t_corr = t + b/H` (`b` = bottom-layer hit position, `H` =
assumed height) — the intra-detector analogue of two-detector parallax
focusing, at the scale of the aperture rather than the stereo baseline.
`python scripts/refocus_analysis.py` → `reports/refocus_analysis.{png,json}`
establishes:

- **They really are that shear**, not a metric rescaling: `E[t_corr]·H −
  E[t_∞]·H = E[b]` is constant to 0.25 mm (0.1934 m) across H = 1…10 m, in all
  three files and both axes. An earlier `selfcal.py` docstring claimed the
  opposite; it is corrected.
- **Focusing is real but weak**: both focus metrics prefer the
  independently-known 7.0 m ceiling, but vary only **~7%** over 5–10 m while
  collapsing 5× by H = 1. A 400-replica Poisson bootstrap puts the peak at 7 m in
  **66%** of replicas (signal-band power) / **53%** (beam modulation), with most of
  the rest at 10 m — a preference, not a measurement. It excludes a low ceiling
  decisively and cannot localize a high one. The flatness is geometric — residual
  blur is `A·|1/H − 1/H_true|` — so no choice of metric recovers sensitivity.
- **Which focus metric**: conventional sharpness (gradient energy, Laplacian
  variance, total variation) *fails* here — the maps are noise-dominated at high
  frequency (Poisson floor ≈ 3× the signal power) and the shear smooths noise with
  signal, so raw high-frequency measures track the shear amount and rise
  monotonically with H; all four tested peak at the scan edge. Two work:
  *beam modulation depth* (narrowband at the known pitch — scene-specific), and
  **`focus.spectral_focus` / `focus.signal_band`** — subtract the analytic Poisson
  floor from the 2-D power spectrum and integrate over the band where the scene
  exceeds it, detected from the data. The latter presumes nothing about the scene,
  and separates refocused from un-refocused by 16% versus 1%.
- **The practical gain is small**: beam FWHM 0.534 → 0.511 m (4.4%). The blur is
  set not by the aperture but by the spread of `b` within one angular bin, which
  the four-layer coincidence makes several times narrower.
- **It must not be fed to the reconstruction.** Since `t_corr = X/H`, a refocused
  bin is a source position over H, not a direction, and the forward model's
  `pose + t·(z − pose.z)` coincides with it only at `z = H` exactly. Off that
  plane it is a rescaling about the detector rather than parallax. Measured:
  beam offset 0.100 m → 1.404 m, far past the 0.15 m distrust gate.

This also settles the proposed assume-H → sharpen → re-measure-H iteration: the
sharpening step destroys the very parallax the height scan reads, and there is
only 3% of signal to iterate on. Not worth building for this geometry.

## Uncertainties and systematics

`python -m muontomo.uncertainty --run <run>` propagates the Poisson counting
statistics by parametric bootstrap (resample the raw counts, rerun the layered
solve and the autofocus): per-pixel sigma map of the layer, error bars on each
beam's position and amplitude, and a confidence interval on the autofocused
height (`uncertainty.json` + `images/uncertainty.png`).

`python scripts/mcs_systematics.py` bounds the main unmodelled physics —
multiple Coulomb scattering in the concrete beams (Highland formula,
flux-weighted): blur sigma ≈ 0.08 m at the ceiling, 3.5× below the 0.28 m
angular-bin footprint → ≈6% beam widening / amplitude suppression, beam
positions unbiased (`reports/mcs_systematics.json`).

Note the alias hazard for periodic ceilings: false height solutions repeat
every `pitch × z / baseline` (~5.8 m here — safely out of range, but check it
for any new geometry). The historical "residual periodic dips at the same
tan θx in both positions" puzzle is consistent with this: at the true 7.0 m
height the inter-view parallax (~0.27 tan-units) is close to the beam pitch
projected to tangent space (~0.23), so the periodic pattern nearly re-registers
between the views.

## Geometry self-calibration

Stage A (`muontomo.selfcal.stage_a`) attempts NCC registration between the two
positions' XY back-projection maps at canned heights — **diagnostic only**.
Stage B (`muontomo.selfcal.stage_b`) is authoritative: it minimizes symmetric
cross-position chi2 through the full aperture-aware forward model in a bounded
window around the user-supplied prior; validated on phantom data with an
injected pose error (`tests/test_selfcal.py`). The production pose is
dx≈1.78 m, dy≈0.72 m, yaw≈0.05° (`configs/production.json`).

**Open item — the aperture is wrong, and fixing it is not a config edit.** The
hit-position histograms are a hard-edged flat top: 23 bars × 1.538 cm =
**0.354 m** active width, against `aperture_m = 0.65` in the config. This
survived because the angular acceptance constrains only the *ratio*
`aperture_m / detector_height_m` (0.65/0.8 = 0.81 vs a measured cutoff of 0.86),
while the absolute value sets the forward model's blur. Correcting it alone
improves χ² 2.086 → 1.964, CNR 1.261 → 1.407 and noise 0.185 → 0.147, but pushes
the beam offset to 0.236 m; re-running Stage B on top moves pos1 by 0.119 m in
both axes and makes it worse still (offset 0.416 m, noise 0.328) even as CNR
leaps to 3.18 — crisp beams in the wrong place. Stage B's cross-position χ²
objective will buy fit quality with a pose shift, so aperture, pose and height
are entangled; correcting the aperture requires re-deriving the pose against an
objective that includes the model-free beam positions. Production is
deliberately left on the old value until then.

## Enhancement suite (`muontomo/enhance/`)

Modular post-reconstruction enhancers, all gated by the beam verification so
nothing can move or invent beams. One shared data/calibration front-end
(`enhance/context.py`); adding a technique = one module + one registry line.

| method | idea | needs |
| --- | --- | --- |
| `guided` | guided filter, edges from the measured backprojection | scipy |
| `dip` | Deep Image Prior: untrained U-Net, data-consistent chi2, early stopping on held-out bins | torch |
| `pnp` | plug-and-play NLM denoiser inside SIRT | scikit-image |
| `clean` | artifact removal: support taper (kills the limited-angle boundary ring) + guided denoise + floor suppression + direction-aware sharpen | scipy |
| `dipclean` | `clean`'s taper + floor subtraction on top of the DIP layer — best beam positions (0.04 m) *and* artifact removal | torch (or a precomputed `enhance/dip.npy`) |

`python -m muontomo.enhance --run runs/production --method all` writes per-
method layers, metrics, PNGs and a comparison `enhance/REPORT.md`. On the
production run: `dipclean` is the best surface (offset 0.04 m, beam CNR 4.1);
all methods PASS the gate. Each enhanced layer appears as a labeled surface in
the viewer.

## Viewer

`python -m muontomo.viewer.build --run <run>` produces a self-contained
`viewer.html` (no network): height-relief terrain of any surface
(reconstruction / DIP / clean / dipclean / single-detector / measured data),
full-room voxel toggle, all-z voxel cloud with density cutoff and z-cut,
iso-surfaces, verified-beam guides, autofocus HUD line, palette selector,
collapsible panel, PNG export.

## Tests

```
pytest tests/ -q
```

68 tests cover calibration math, raycasting (adjoint identity, path lengths),
full phantom round-trips per algorithm, every metric on hand-computed
fixtures, self-calibration recovery of an injected pose error, the autofocus
(NCC curve, height map, two-stage CV scan, real CV scorer on a tiny phantom),
the bootstrap uncertainty propagation, the refocus edge-shift correction and
the joint triangulation least-squares (including that it declines to guess with
a single pose), the artifact-cleanup enhancers and their anti-hallucination
gate, the autofocus PDF report, and the viewer (build + Playwright smoke test). The
suite runs without torch/scikit-image — CI (`.github/workflows/tests.yml`)
verifies exactly that configuration.

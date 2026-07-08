# Muon Tomography: 3D Room Reconstruction

Reconstructs a 3D density-length volume of a cafeteria room's ceiling from
cosmic-ray muon flux measured at two detector positions, calibrated against a
clear-sky reference. See `detector_summary.md` for the detector itself.

## Data

`data/` holds three ROOT files of pre-made histograms (no event trees):

- `HistsOutSkyRoofRuns37-77.root` — clear-sky reference (28.5M tracks)
- `HistsOutDataCafePos0.root`, `HistsOutDataCafePos1.root` — detector inside
  the cafeteria at two positions ~1.5-2 m apart, horizontal movement only,
  same floor (6.3M / 2.4M tracks)

The angular flux map `txty` (tan θx vs tan θy, 800×800 bins over ±2) is the
primary analysis histogram; `XY01m..XY10m` are back-projected planes at fixed
heights, useful for diagnostics (see Stage A below) but not physically
independent of `txty` for a single position.

## Install

```
pip install uproot numpy scipy matplotlib pytest playwright
```

## Physics model

For each angular bin at each position: transmission `T = N_cafe / (s·N_sky)`,
opacity `λ = -ln T ≈ ∫κ dl` along that bin's ray. The detector's 65×65 cm
aperture is comparable to the ceiling-beam pitch at ~3 m, so each bin is
modeled as a bundle of parallel sub-rays across the aperture (angle-dependent,
narrowing with the 4-layer coincidence requirement), not a pinhole. With only
two views this is severely limited-angle tomography: expect vertical
elongation; TV regularization and a layered (height-scan) model exist to
counter it. See `muontomo/reconstruct.py` and `muontomo/raycast.py`.

## Pipeline

```
scripts/inspect_data.py        # Stage 0: dump histogram metadata, convention checks
muontomo.selfcal                # geometry self-calibration (Stage A diagnostic, Stage B authoritative)
muontomo.reconstruct            # produce a run: volume.npz + holdout volumes
muontomo.evaluate                # metrics.json scorecard + standard PNG set
muontomo.compare                 # IMPROVED/REGRESSED verdict between two runs
muontomo.phantom                 # synthetic ground truth for rigorous algorithm gating
muontomo.viewer.build            # self-contained viewer.html
muontomo.viewer.smoke            # Playwright check of the viewer
```

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
5. python -m muontomo.compare runs/prev runs/expNN
6. algorithm changes additionally gated on phantoms (real ground truth):
     python -m muontomo.phantom --preset beams --seed 42 --out phantoms/p1
     python -m muontomo.reconstruct --config <cfg pointed at phantoms/p1> --out runs/ph_expNN
     python -m muontomo.evaluate --run runs/ph_expNN --truth phantoms/p1/truth_volume.npz
7. converged -> python -m muontomo.viewer.build --run runs/expNN
              -> python -m muontomo.viewer.smoke --run runs/expNN
              -> inspect runs/expNN/images/viewer_*.png
```

**Metric roles**: phantom-truth metrics (RMSE/SSIM/IoU/z-error) gate algorithm
changes — they're the only metrics with real ground truth. Fidelity and
cross-position-CV metrics gate calibration/geometry changes. Structural
metrics (periodicity, contrast) are advisory only — always computed on the
*measured* maps too, so scores read as a fraction of what the data itself
supports, since a strong prior can otherwise game them.

Every `runs/<name>/` is reproducible: `config.json` snapshots the full config
+ git hash; `evaluate` only ever adds `metrics.json` and `images/`, never
touches the volume.

## Geometry self-calibration

Stage A (`muontomo.selfcal.stage_a`) attempts NCC registration between the two
positions' XY back-projection maps at each canned height — **diagnostic
only**. On this campaign the expected parallax shift (baseline/height, ~0.5
tan-units for a ~1.75 m baseline at a ~3 m ceiling) is comparable to the whole
angular acceptance, so the shifted overlap is small and the NCC curve comes
out flat and noisy; it cannot reliably resolve the offset by itself.

Stage B (`muontomo.selfcal.stage_b`) is authoritative: it minimizes symmetric
cross-position chi2 (reconstruct from one position, score against the other,
through the full aperture-aware forward model) in a bounded window around the
user-supplied prior. Validated on `phantom.py` data with an injected pose
error (`tests/test_selfcal.py`). On the real data it refined the prior
(dx=1.75 m) to dx≈1.96 m, dy≈0.34 m, yaw≈-0.05°
(`configs/production.json`).

## Open finding: residual periodic dips

`runs/production/images/panel_summary.png`'s residual-pull maps show periodic
opacity dips at nearly the **same** tan(θx) in both positions, despite the
~2 m baseline — the naive parallax-shift expectation (~0.5 tan-units) isn't
seen. Two explanations remain open: (1) the offset is close to an integer
multiple of the true beam pitch, aliasing the pattern near zero shift, or (2)
part of the periodic structure is a detector-referenced (angle-fixed)
systematic rather than purely the ceiling. This is exactly the kind of
question the evaluation harness is built to surface — investigate via
`muontomo.selfcal.depth_from_parallax` curves, per-channel occupancy
(`scripts/inspect_data.py`), or a finer phantom study with a controlled
pitch/baseline ratio.

## ML

No ML in the baseline: with 2 views and O(10^3) usable bins vs O(10^4-10^5)
voxels, the prior (nonnegativity + anisotropic TV + layered model) dominates
and stays debuggable. The one justified next experiment is **Deep Image
Prior** (untrained CNN reparameterization) behind the same `Reconstructor`
protocol in `reconstruct.py`, accepted only if it beats TV on phantom RMSE
*and* real-data cross-validation — the random-bin-holdout chi2 already
computed in `evaluate.py` solves DIP's usual early-stopping problem. A
regularizer trained on synthetic beam phantoms is not recommended: it would
hallucinate periodic beams, and the periodicity metric would reward exactly
that hallucination.

## Tests

```
pytest tests/ -q
```

40+ tests cover calibration math, raycasting (adjoint identity, path lengths),
full phantom round-trips per algorithm, every metric on hand-computed
fixtures, self-calibration recovery of an injected pose error, and the viewer
(build + Playwright smoke test).

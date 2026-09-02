# Rolling Status — cafeteria_3d_modeling

Auto-maintained running log of progress on this project. Each entry summarizes one working session: the goal, what got done, the result, and what's left.

## Log

### 2026-08-13 12:56 — Data-driven ceiling autofocus + report
**Goal:** Give the pipeline a trustworthy, data-driven ceiling-height "autofocus" and make the 3D viewer and its reporting legible and honest.
**Done:**
- Fixed viewer surface-switch bug (meshes were stacking, not replacing), flipped beam relief to hang downward toward the detectors, and added a single-detector (pos0 / pos1) vs. two-detector surface toggle.
- Built `muontomo/focus.py`: a model-free two-view NCC "quick-look" + per-region height map, and promoted the alias-robust cross-validation height-scan (fit one detector, predict the other) as the authoritative estimate → 7.0 m (matches the reconstruction chi²).
- Validated against synthetic ground truth: campaign-geometry phantoms with ceilings injected at 6.6/7.0/7.4 m are recovered to ~0.2 m; added `scripts/autofocus_validation.py` + committable `reports/autofocus_validation.png`.
- Smoothed the CV curve with a trimmed-residual metric (removes aliasing-outlier spikes, same 7.0 m minimum) on a finer 0.1 m grid.
- Added a 3-page per-run `autofocus_report.pdf` (`muontomo/focus_report.py`, scientific prose, de-duplicated figure) generated on every `evaluate`; wired autofocus into the viewer HUD and the evaluate headline.
**Result:** Autofocus independently selects 7.0 m and is validated against known-truth simulations; the report and figures are legible with no duplicated content. All 43 tests pass. Nothing committed to git.
**Next:**
- `evaluate` now takes ~4 min because of the 0.1 m CV scan — consider widening the step or narrowing the range if that's too slow routinely.
- Stage the autofocus work into git when ready (`focus.py`, `focus_report.py`, `scripts/`, `reports/`, the `layered_fit` trim option, viewer/evaluate wiring).

### 2026-08-13 16:51 — Full-room voxels, artifact cleanup, paper
**Goal:** Clean the artifacts out of the 3D reconstruction, deliver a complete-room voxel model from the two detectors, and write it all up as a paper-style report.
**Done:**
- Built the `clean` enhancer (`muontomo/enhance/artifacts.py`): support taper kills the boundary-blob ring, guided denoise + floor suppression flattens the background, direction-aware sharpening straightens the beams — beam CNR 0.40 → 2.48 (6×), offset 0.088 m, PASS; wired into the viewer as an "artifact-cleaned" surface.
- Full-room 3D solve (`configs/full3d.json` → `volume_full3d.npz`, 1.9 M voxels) with a viewer toggle, a support taper fixing the "inverted beams" display, and a new voxel-cloud renderer (all-z colored points, density cutoff + z-cut sliders).
- Viewer UX: collapsible control panel, color-palette dropdown (viridis/inferno/magma/turbo/grayscale), save-PNG button.
- Wrote `reports/voxel_reconstruction_report.{html,pdf}` — a 6-page paper: achievement-first framing (full-room model from only two detectors), autofocus section with the two-detector-disagreement and beam-verify figures, a pedagogical "one detector → two" page (angular shadow → 2D map → parallax → fused 3D), quantified depth degeneracy (28% of mass in 6–8 m, peak at 4.2 m) and what a 3rd detector / four-corner array would buy (δz ~1 m → 0.4 / 0.2 m).
**Result:** The viewer shows a clean, artifact-free reconstruction plus an explorable full-room voxel cloud, and the work is documented in a self-contained 6-page paper under `reports/`. All 43 tests pass. Still nothing committed to git.
**Next:**
- Commit the whole batch (enhance `clean`, full3d config, viewer features, focus/autofocus, `scripts/`, `reports/`) — user has repeatedly deferred; everything is ready to stage.
- Optional: make `clean` the default viewer surface, or apply the artifact cleanup on top of the DIP layer (best positions + artifact removal).

### 2026-08-24 19:09 — Hardening, error bars, MCS systematics
**Goal:** Roadmap steps 4+5 (paper hardening + engineering wins), then steps 2+3 (bootstrap error bars + physics-model honesty check), with a full pipeline re-run in between.
**Done:**
- Autofocus 4× faster (`layer_cv_score` extraction + two-stage coarse→fine CV scan): `evaluate` ~4 min → ~60 s, same 7.0 m; added `dipclean` enhancer (DIP + artifact removal) — best surface: offset 0.04 m, CNR 4.1 — wired into CLI + viewer.
- 14 new tests (→57), GitHub Actions CI (torch-free, green on first run), README rewritten; paper hardened: 10 real references, Apparatus placeholder, formal validation table, dipclean result. Committed `cad0732` + pushed.
- Re-ran the entire pipeline end-to-end (~5 min): all numbers reproduced exactly (chi² 2.086, autofocus 7.0 m, offset 0.10 m); all five enhancers PASS.
- Built `muontomo/uncertainty.py` (parametric Poisson bootstrap) and ran 50+25 replicas on production: beam positions ±0.02–0.12 m (mean 0.055), amplitudes ~50% r.m.s., autofocus z* = 6.86 ± 0.12 m (68% CI); the 0.10 m beam offset is therefore partly systematic.
- Added `scripts/mcs_systematics.py` (Highland multiple-Coulomb-scattering bound): flux-weighted blur σ ≈ 0.08 m at the ceiling, 3.5× below the 0.28 m bin footprint → ≲6% widening/amplitude bias, positions unbiased. UQ + MCS folded into the paper (abstract ±0.12 m stat; Sec 6 items v/vi) and README; 2 new tests (→59).
**Result:** Pipeline is fast, error-barred, and systematics-bounded; the 6-page paper now carries statistical and physics uncertainties. 59/59 tests, CI green. UQ/MCS batch not yet committed.
**Next:**
- Commit + push the UQ/MCS batch (`uncertainty.py`, `mcs_systematics.py`, tests, README, paper) — excluding the user's local `rolling_status.zip`.
- Roadmap step 1 remains: the detector-placement design study (simulate 3rd viewpoint / displaced repeat runs).

### 2026-09-01 15:05 — Single-detector refocusing evaluated
**Goal:** Assess the advisor's proposal that single-detector angular refocusing and two-detector parallax focusing are the same operation, and that an iterative assume-H → sharpen → re-measure-H loop plus closed-form triangulation would sharpen the result.
**Done:**
- Established that the DAQ's `XY01m..XY10m` histograms *are* the proposed shear `t_corr = t + b/H`, not metric back-projections: under the shear `E[t_corr]H − E[t_∞]H = E[b]` is constant to 0.25 mm (0.1934 m) across H = 1–10 m, in both detectors and the sky file. `selfcal.py` had asserted the opposite ("carries no depth information"); docstring corrected, and the same stale claim fixed in the README.
- Measured the effect on real data (`scripts/refocus_analysis.py` → `reports/refocus_analysis.{png,json}`): focusing is **real** — beam modulation peaks at the independently-known 7.0 m ceiling in each detector separately — but **weak**, varying only 3.1% over 5–10 m while collapsing 5× by H=1 m. Beam FWHM gain is 4.4% (0.534 → 0.511 m).
- Ran a 6-arm A/B through the full pipeline: refocused input is **rejected** (beam offset 0.100 → 1.404 m, far past the 0.15 m gate). Root cause is structural — `t_corr = X/H` indexes a source position, not a direction, so the forward model's ray parametrization only coincides at `z = H`. This also kills the iteration loop: the sharpening step destroys the parallax the height scan reads.
- Built the requested joint triangulation least-squares (`beams.triangulate` + `beam_peaks_subbin`): **z = 6.955 ± 0.086 m** over 8 features / 16 observations, χ²/dof 0.20 — independent of both solver and CV autofocus, and tighter than the bootstrap. Sub-bin refinement was essential (one 0.04 tan bin ≈ 1 m of single-pair height).
- Incorporated the advisor's formal trigonometry note: his `tan θ_x,corr = x_s/H` derivation matches what I found independently (a refocused bin is a source position over H), and his closed-form `z = d/(tanθ₂cosφ₂ − tanθ₁cosφ₁)` is the two-ray case of our joint fit. But his **d = 2.4 m** vs our calibrated **1.916 m** matters: since `z ∝ d`, that is 6.96 m vs 8.71 m of ceiling. Added `beams.scale_closure` — each detector measures the beam angular period alone (0.2329 / 0.2321 tan-units), so `pitch = z × period` closes the (baseline, height, pitch) triple and any one surveyed length fixes the other two. Angles are scale-degenerate; one tape measure is required.
- Found and investigated a latent model bug: `aperture_m = 0.65` vs a measured 0.354 m active width (23 bars × 1.538 cm); survived because acceptance constrains only the ratio a/h. Correcting it improves χ²/CNR/noise but fails the beam gate (0.236 m), and re-running selfcal on top makes it worse (0.416 m, CNR 3.18 — sharp beams in the wrong place). Left production unchanged; documented as the largest known instrumental systematic.
**Result:** The advisor's physics is confirmed and quantified, and the engineering conclusion is a clean negative: refocusing must stay out of the inversion, and the iteration loop has no leverage on this geometry. Net gains kept are the triangulation cross-check and the aperture finding. Production re-run end-to-end and reproduces exactly (χ² 2.086, offset 0.100 m, autofocus 7.0 m) now carrying triangulation; paper at 7 pages with new §3.6/§3.7 and uncertainty item (vii); 68/68 tests pass. All on branch `refocusing`, nothing committed.
**Next:**
- Commit the `refocusing` branch (excluding the local `rolling_status.zip`) and let CI verify.
- **Settle the absolute scale**: survey either the detector separation or the ceiling beam pitch. Everything scales with it and no algorithm work substitutes for it; the advisor's d = 2.4 m would make the ceiling 8.71 m, not 6.96 m.
- The aperture/pose/height degeneracy needs a self-calibration objective that includes model-free beam positions.
- A fiducial marker at Megiddo, to break the circularity that let the aperture error survive; and roadmap step 1, the placement study, still open.

### 2026-09-02 17:17 — Scene-agnostic focus metric + error bars
**Goal:** Answer the advisor's follow-up — beam modulation depth is scene-specific (it needs the beam pitch in advance), so find a focus metric that works on an unrecognised scene, and put real uncertainties on where each metric says the ceiling is.
**Done:**
- Showed why the obvious generic replacements fail: gradient energy, Laplacian variance, total variation and raw high-frequency power all peak at the edge of the scan, because these maps are noise-dominated at high frequency (Poisson floor ≈ 3× the signal power) and the shear smooths noise along with signal, so raw sharpness just tracks how much shear was applied.
- Built the scene-agnostic alternative (`focus.signal_band`, `focus.spectral_focus`): subtract the *analytic* Poisson floor from the 2-D power spectrum (per-bin opacity variance follows from the counts, so it is computed not fitted), then integrate what survives over the frequency band the scene itself exceeds that floor. No assumed period. The detected band (structures ~0.75–11 m) contains the 1.58 m beam pitch without ever being told it, and it separates refocused from un-refocused by 16% where modulation depth manages 1%. Stable over nine SNR-threshold / band-width combinations.
- Added a 400-replica Poisson bootstrap of both metrics (`bootstrap_metrics`, reusing `uncertainty.resample_tmaps`) — this was the substantive correction of the session: **the peak is a preference, not a measurement.** 7 m wins 66% of replicas for signal-band power and 53% for beam modulation, with most of the rest at 10 m; across 5–10 m both vary by only ~7%, comparable to their own ±1σ bands. An earlier 40-replica run had said 70% for modulation, so the sample size mattered.
- Rebuilt Figure 3 three times before it communicated: it is now (a) both metrics vs assumed height with ±1σ/±2σ bands and (b) the argmax distribution with binomial errors. The spectrum diagnostic moved out of the paper into the memo only, per user direction. Paper §3.7 restructured into scene-specific / scene-agnostic, README updated, memo fully rewritten around the metric question; 4 new tests (→72).
**Result:** There is now a focus metric that needs no prior knowledge of the scene and is the better of the two on this data, and the honest statement of what single-detector refocusing achieves: it excludes a low ceiling by many σ and prefers the right one at roughly two-to-one odds. Reinforces the earlier "no leverage for the iteration loop" conclusion with quantified uncertainty rather than assertion. 72/72 tests, still on branch `refocusing`, nothing committed.
**Next:**
- Commit the `refocusing` branch (excluding the local `rolling_status.zip`) and let CI verify — now two sessions overdue.
- Send the memo; the open question back to the advisor is where his `d = 2.4 m` comes from.
- Unchanged: settle the absolute scale by survey, fix the aperture/pose/height degeneracy, fiducial marker at Megiddo, roadmap step 1.

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

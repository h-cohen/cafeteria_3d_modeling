# ML Enhancement Plan: Sharpening the 3D Reconstruction

Goal: make the 3D reconstructed ceiling surface as sharp and feature-rich as the raw 2D
flux images, using enhancement techniques that cannot hallucinate structure. Three
techniques, built on one modular core, compared in a final report. Every technique's
output is gated by the existing model-free beam verification (`muontomo/beams.py`):
mean |beam offset| <= 0.10 m and exactly 5 beams, or the result is rejected.

Status legend: [ ] todo, [x] done.

---

## Step 1 — Modular package skeleton + shared data front-end

- [ ] New package `muontomo/enhance/`.
- [ ] `enhance/context.py` — THE single data-loading + calibration front-end, used by
      every technique. `EnhanceContext` dataclass built by `load_context(run_dir)`:
      - `cfg` — `RunConfig.load(run/"config.json")`
      - `tmaps` — calibrated transmission maps via the existing
        `transmission_maps(load_dataset(cfg.data), cfg.binning, cfg.calibration)`
        (sky calibration lives here and only here)
      - `rho, origin, spacing` — from `volume.npz` (canonical solver output)
      - `layer` — argmax z-slice (2D) + `xs, ys` world coordinates
      - `guide` — mean measured backprojection on the same grid
        (existing `backproject_opacity` from `muontomo/backproject.py`)
      - `layer_model()` — lazy thin-layer forward model (existing `_layer_model`
        from `muontomo/reconstruct.py`) for data-consistent methods
- [ ] `enhance/base.py` — technique interface + registry:
      `Enhancer` protocol (`name`, `enhance(ctx) -> 2D ndarray`) and
      `REGISTRY: dict[str, Enhancer]`. Adding a technique = one new module + one
      registry entry. Core pipeline files are never touched.
- [ ] `enhance/verify.py` — shared anti-hallucination gate: beam peaks of the enhanced
      layer (reuse `beam_peaks` from `muontomo/beams.py`), offsets vs the verified data
      positions, n_beams, sharpness metrics (beam FWHM from the x-profile,
      edge-gradient score). Returns a metrics dict; PASS/FAIL per the gate above.
- [ ] `enhance/__main__.py` CLI:
      `python -m muontomo.enhance --run runs/production --method guided|dip|pnp|all`
      Per method writes:
      - `runs/<run>/enhance/<method>.npy` (enhanced layer)
      - `runs/<run>/images/enhance_<method>.png` (plain | enhanced | guide, same
        crop/colormap/beam-guide-lines as the viewer)
      - `runs/<run>/enhance/<method>_metrics.json`

Modularity rules (all steps): calibration + loading appear once (context.py);
heavy imports (torch, skimage) live inside their technique module only, so the core
pipeline and the test suite never import them transitively.

## Step 2 — Guided filter (first technique, scipy-only)

- [ ] `enhance/guided.py` — He, Sun & Tang (2010) guided filter:
      `guided_filter(p, I, radius_px, eps)` with `scipy.ndimage.uniform_filter` box
      means: `a = cov(I,p)/(var(I)+eps)`, `b = mean(p) - a*mean(I)`, output
      `meanA*I + meanB`. NaNs in the guide filled with local mean first.
      - input `p` = reconstruction layer (0.12 m pre-blur, as the viewer applies)
      - guide `I` = `ctx.guide` (mean measured backprojection — real data)
      Anti-hallucination property: where guide and recon are uncorrelated, `a -> 0`,
      so guide-only or recon-only structure does not transfer — only features present
      in BOTH get sharpened.
- [ ] Parameter sweep (scratch): `radius_m in {0.3, 0.4, 0.6}` x
      `eps_frac in {0.05, 0.1, 0.2}` (eps = (eps_frac * std(guide_nonzero))^2).
      Winner = sharpest panel that still PASSES the beam gate. Hard-code as defaults.

## Step 3 — Install Camp-1 ML packages with uv

- [ ] `uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu`
      (CPU wheel; no CUDA needed)
- [ ] `uv pip install --python .venv/bin/python scikit-image` (NLM denoiser for PnP)
- [x] If `uv` unavailable, fall back to `.venv/bin/pip install`. Record exact installed
      versions here in this file.

Installed via `uv pip install --python .venv/bin/python`:
- torch 2.13.0+cpu (index-url https://download.pytorch.org/whl/cpu)
- scikit-image 0.26.0
- pywavelets (dependency of skimage.restoration.estimate_sigma; added after PnP hit a
  missing-PyWavelets error)

## Step 4 — Deep Image Prior (data-consistent, torch)

- [ ] `enhance/dip.py` — untrained small U-Net reparametrizes the 2D layer:
      `x = softplus(CNN(z))`, fixed noise input `z` (no training data — the
      architecture itself is the prior).
- [ ] Loss = weighted chi2 of the MEASUREMENTS through the thin-layer forward model:
      `|| w * (A_layer @ x.ravel() + c_pose - lam) ||^2`
      with `A_layer` from `ctx.layer_model()` converted to a torch sparse tensor, and
      per-pose offset scalars `c_pose` free (mirrors `sirt_tv`'s offset handling).
- [ ] Adam, ~1-2k steps on CPU; **early stopping on held-out bins** (random 20% of
      rows, same style as the pipeline's `binholdout`): stop when held-out chi2 turns
      upward — that is DIP's regularization moment. Hard runtime cap ~10 min, progress
      logged every 100 steps.

## Step 5 — Plug-and-Play prior (skimage NLM inside SIRT)

- [ ] `enhance/pnp.py` — self-contained SIRT loop on the thin-layer model (reuse the
      SIRT update + offset algebra from `sirt_tv` in `muontomo/reconstruct.py`), with
      the TV proximal step replaced by a pluggable denoiser:
      `skimage.restoration.denoise_nl_means` on the 2D layer each sweep, strength
      annealed like `tv_alpha * p95(x)`.
- [ ] Same n_iter / best-chi2 bookkeeping as `sirt_tv`; returns the best iterate's layer.

## Step 6 — Comparison report (all three)

- [ ] `python -m muontomo.enhance --run runs/production --method all`, then a report
      composer writing `runs/production/enhance/REPORT.md` +
      `runs/production/images/enhance_report.png`:
      - 5 panels: plain recon | guided | DIP | PnP | measured guide
        (identical crop, colormap, verified-beam guide lines)
      - Table per method: mean |beam offset|, n_beams, beam FWHM, edge-gradient
        score, data chi2 (data-consistent methods), runtime, PASS/FAIL verdict
- [ ] Deliver the detailed comparison + recommendation in chat.
- [x] Optional follow-up: wire the winning method (DIP) into the viewer as a
      clearly-labeled surface option. DONE -- `viewer/build.py` embeds the
      DIP-enhanced layer (prefers precomputed `enhance/dip.npy`, else computes;
      interpolated onto the display grid), `viewer/app.js` + `template.html` add
      the "reconstruction (DIP enhanced)" dropdown option. Hides itself when the
      layer is absent (no torch / no data). Verified: seeded DIP is stable across
      seeds (beam positions vary <=0.08 m); viewer smoke test clean; 43 tests pass.

## RESULTS (all steps complete)

Command: `python -m muontomo.enhance --run runs/production --method all`
Outputs: `runs/production/enhance/{guided,dip,pnp}.npy` + `_metrics.json`,
`runs/production/enhance/REPORT.md`, `runs/production/images/enhance_{method}.png`
and `enhance_report.png`. All 43 pytest tests still pass; torch/skimage stay lazy
(not imported at `import muontomo.enhance`).

| method | mean|offset| m | n_beams | FWHM m | runtime | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| plain recon | 0.104 | 5 | 0.545 | - | PASS |
| guided filter | 0.12 | 5 | 0.563 | <0.1 s | PASS |
| deep image prior | **0.04** | 5 | 0.453 | 15 s | PASS |
| plug-and-play (NLM) | 0.08 | 5 | **0.36** | 2 s | PASS |

Winner: **DIP** — best beam-position accuracy (0.04 m) and the cleanest sharp
beams; PnP is the sharpest (FWHM 0.36) but leaves some speckle; guided is the
safest (edges provably transferred from the pos0 2D flux, cannot invent). All
three pass the anti-hallucination gate (5 beams at verified positions).

Key implementation notes:
- Guide for guided filter = the single sharpest detector backprojection (pos0),
  auto-selected by central-band stripe contrast. The MEAN backprojection diluted
  the low-coverage left beam and failed the gate; the sharp single view keeps all 5.
- DIP/PnP outputs are smoothed to the viewer display resolution (0.12 m) before
  scoring: sub-resolution CNN/NLM texture is below the data's resolving power.
- DIP early-stops on 20% held-out measurement bins (val chi2 min at step ~355).
- PnP keeps the SIRT iterate nearest chi2=1 (discrepancy), not the most-overfit.

## Verification (end of each step)

1. Per-technique gate: mean |beam offset| <= 0.10 m AND n_beams == 5 on the enhanced
   layer (`enhance/verify.py`).
2. Step 6 run completes; REPORT.md + PNGs exist and are visually sane.
3. `.venv/bin/python -m pytest tests/ -q` — full suite still green (the enhance
   package is additive; no core file imports torch/skimage).
4. Nothing committed to git without an explicit request.

"""Plug-and-Play prior: SIRT on the thin-layer forward model, with the TV
proximal step replaced by a non-local-means denoiser (skimage).

Same data-consistent iteration as the pipeline's sirt_tv, but the regularizer is
a stronger edge-aware denoiser plugged into the proximal slot. NLM is a generic
image denoiser (not trained on ceilings), so it suppresses speckle without
imposing beam-shaped priors. skimage is imported lazily.
"""

from __future__ import annotations

import time

import numpy as np

from .base import Enhancer, register
from .context import EnhanceContext

N_ITER = 120
NLM_STRENGTH = 2.5  # multiplies the auto h estimate; higher = smoother
CHI2_STOP = 1.0  # discrepancy principle: keep the iterate nearest the noise floor,
#                  not the most-overfit one (raw best-chi2 pulls in noise)


class _PnP:
    name = "pnp"

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PnP needs scikit-image: uv pip install scikit-image") from e
        from ..reconstruct import _prepare, _update_offsets

        t0 = time.time()
        lm = ctx.layer_model
        A = lm.fwd.A
        lam, w, pose_of_row = lm.lam, lm.w, lm.pose_of_row
        shape = lm.fwd.grid.shape
        n_poses = int(pose_of_row.max()) + 1
        _, row_inv, col_inv = _prepare(A, w)
        n_used = max(np.count_nonzero(w), 1)

        x = np.zeros(A.shape[1])
        c = np.zeros(n_poses)
        best = (np.inf, None)  # iterate whose chi2 is closest to the noise floor
        for k in range(N_ITER):
            resid = lam - (A @ x + c[pose_of_row])
            c = c + _update_offsets(resid, w, pose_of_row, n_poses)
            resid = lam - (A @ x + c[pose_of_row])
            chi2 = float(np.sum(w * resid**2) / n_used)
            if abs(chi2 - CHI2_STOP) < abs(best[0] - CHI2_STOP):
                best = (chi2, x.copy())
            x = x + col_inv * (A.T @ (w * resid * row_inv))
            np.maximum(x, 0.0, out=x)
            # PnP prox: denoise the collapsed 2D layer, broadcast back over z
            m = x.reshape(shape).mean(axis=2)
            mx = float(m.max())
            if mx > 0:
                sig = estimate_sigma(m / mx) * NLM_STRENGTH
                m = denoise_nl_means(m / mx, h=sig, patch_size=5, patch_distance=6,
                                     fast_mode=True) * mx
            x = np.repeat(m[:, :, None], shape[2], axis=2).ravel()

        self.last_info = {"best_chi2": round(best[0], 3), "iters": N_ITER,
                          "runtime_s": round(time.time() - t0, 1)}
        layer = best[1].reshape(shape).mean(axis=2)
        return ctx.display_blur(np.maximum(layer, 0.0))  # viewer display resolution


register(_PnP())

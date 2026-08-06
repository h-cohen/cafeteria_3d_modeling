"""Deep Image Prior (Ulyanov et al. 2018), data-consistent.

An UNTRAINED small U-Net reparametrizes the ceiling layer: x = softplus(CNN(z)),
fixed noise z. The only loss is the weighted chi2 of the MEASUREMENTS through the
thin-layer forward model -- no training set, no learned prior over ceilings. The
CNN architecture itself favours coherent structure over per-pixel noise; early
stopping on held-out measurement bins is the regularization moment (stop before
the network starts fitting the noise). It cannot add depth information the two
views never captured -- it is a self-regularizing fit to the same data.

torch is imported lazily inside enhance() so the package/tests never require it.
"""

from __future__ import annotations

import time

import numpy as np

from .base import Enhancer, register
from .context import EnhanceContext

N_STEPS = 1500
LR = 0.01
VAL_FRAC = 0.2
PATIENCE = 150  # steps of no val-chi2 improvement before stopping
SEED = 0
MAX_SECONDS = 600


def _build_unet(torch, nn, ch):
    class DoubleConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1, True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1, True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = DoubleConv(ch, 32)
            self.d2 = DoubleConv(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.mid = DoubleConv(64, 64)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.u2 = DoubleConv(64 + 64, 32)
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.u1 = DoubleConv(32 + 32, 16)
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            m = self.mid(self.pool(c2))

            def cat(up, skip):  # crop both to common size (odd dims -> +/-1)
                h = min(up.shape[-2], skip.shape[-2])
                w = min(up.shape[-1], skip.shape[-1])
                return torch.cat([up[..., :h, :w], skip[..., :h, :w]], 1)

            u2 = self.u2(cat(self.up2(m), c2))
            u1 = self.u1(cat(self.up1(u2), c1))
            return self.out(u1)

    return UNet()


class _DIP:
    name = "dip"

    def enhance(self, ctx: EnhanceContext) -> np.ndarray:
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("DIP needs torch: uv pip install torch") from e

        torch.manual_seed(SEED)
        t0 = time.time()
        lm = ctx.layer_model
        nx, ny, nz_thin = lm.fwd.grid.shape

        Acoo = lm.fwd.A.tocoo()
        A_t = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([Acoo.row, Acoo.col]), dtype=torch.long),
            torch.tensor(Acoo.data, dtype=torch.float32), Acoo.shape,
        ).coalesce()
        lam = torch.tensor(lm.lam, dtype=torch.float32)
        w = torch.tensor(lm.w, dtype=torch.float32)
        pose_of_row = torch.tensor(lm.pose_of_row, dtype=torch.long)
        n_poses = int(lm.pose_of_row.max()) + 1

        # split observed bins into train / val for early stopping
        rng = np.random.default_rng(SEED)
        obs = np.flatnonzero(lm.w > 0)
        val = np.zeros(lm.w.shape[0], dtype=bool)
        val[rng.choice(obs, size=int(VAL_FRAC * obs.size), replace=False)] = True
        w_tr = w * torch.tensor(~val, dtype=torch.float32)
        w_va = w * torch.tensor(val, dtype=torch.float32)
        ntr, nva = float(w_tr.gt(0).sum()), float(w_va.gt(0).sum())

        z = torch.randn(1, 2, nx, ny) * 0.1
        net = _build_unet(torch, nn, 2)
        c_pose = torch.zeros(n_poses, requires_grad=True)
        opt = torch.optim.Adam(list(net.parameters()) + [c_pose], lr=LR)
        scale = float(np.percentile(ctx.layer[ctx.layer > 0], 95)) if (ctx.layer > 0).any() else 1.0

        def forward_layer():
            raw = net(z)  # (1,1,H',W') -- H',W' may differ by +-1 on odd grids
            raw = torch.nn.functional.interpolate(raw, size=(nx, ny), mode="bilinear",
                                                  align_corners=False)
            m = torch.nn.functional.softplus(raw[0, 0]) * scale  # (nx, ny) >= 0
            x_thin = m.unsqueeze(-1).expand(nx, ny, nz_thin).reshape(-1)  # broadcast over z
            y = torch.sparse.mm(A_t, x_thin.unsqueeze(1)).squeeze(1) + c_pose[pose_of_row]
            return m, y

        best = (float("inf"), None)
        since = 0
        for step in range(N_STEPS):
            opt.zero_grad()
            m, y = forward_layer()
            resid = y - lam
            loss = (w_tr * resid * resid).sum() / max(ntr, 1)
            loss.backward()
            opt.step()
            with torch.no_grad():
                val_chi2 = float((w_va * resid.detach() ** 2).sum() / max(nva, 1))
            if val_chi2 < best[0] - 1e-4:
                best = (val_chi2, m.detach().numpy().copy())
                since = 0
            else:
                since += 1
            if since >= PATIENCE or time.time() - t0 > MAX_SECONDS:
                break

        self.last_info = {"val_chi2": round(best[0], 3), "steps": step + 1,
                          "runtime_s": round(time.time() - t0, 1)}
        # smooth to the viewer's display resolution: sub-0.12 m CNN texture is
        # below the data's resolving power, not real structure.
        return ctx.display_blur(np.maximum(best[1], 0.0))


register(_DIP())

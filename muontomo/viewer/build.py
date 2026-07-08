"""Build a self-contained viewer.html for a run: no external requests, works
from file:// or as a private artifact.

    python -m muontomo.viewer.build --run runs/exp01
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from ..export import export_volume

VENDOR = Path(__file__).parent / "vendor"
MAX_DIM = 96  # cap embedded volume size; downsample larger grids


def _downsample(rho: np.ndarray, spacing: float, origin) -> tuple[np.ndarray, float, tuple]:
    factor = max(1, int(np.ceil(max(rho.shape) / MAX_DIM)))
    if factor == 1:
        return rho, spacing, origin
    from scipy.ndimage import zoom

    rho2 = zoom(rho, 1.0 / factor, order=1)
    return rho2.astype(np.float32), spacing * factor, origin


def build_viewer(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    run = Path(run_dir)
    export_volume(run)  # writes volume.npy + meta.json (headline metrics, iso hints)
    rho = np.load(run / "volume.npy").astype(np.float32)
    meta = json.loads((run / "meta.json").read_text())

    pos = np.maximum(rho, 0.0)
    pos, spacing, origin = _downsample(pos, meta["spacing_m"], meta["origin_m"])
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0
    scale = 255.0 / max(p99 * 1.5, 1e-9)
    quant = np.clip(pos * scale, 0, 255).astype(np.uint8)
    data_b64 = base64.b64encode(quant.tobytes()).decode("ascii")

    volume_payload = {
        "shape": list(quant.shape),
        "origin_m": list(origin),
        "spacing_m": spacing,
        "value_range": meta["value_range"],
        "suggested_iso": [round(0.3 * 255, 1), round(0.6 * 255, 1)],
        "run": meta["run"],
        "headline_metrics": meta.get("headline_metrics", {}),
        "data_b64": data_b64,
    }

    html = (VENDOR.parent / "template.html").read_text()
    html = html.replace("/*__THREE_JS__*/", (VENDOR / "three.min.js").read_text())
    html = html.replace("/*__ORBIT_CONTROLS_JS__*/", (VENDOR / "OrbitControls.js").read_text())
    html = html.replace("/*__MARCHING_CUBES_JS__*/", (VENDOR / "marchingcubes.js").read_text())
    html = html.replace("/*__VOLUME_JSON__*/", json.dumps(volume_payload))
    html = html.replace("/*__APP_JS__*/", (VENDOR.parent / "app.js").read_text())

    out = Path(out_path) if out_path else run / "viewer.html"
    out.write_text(html)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = build_viewer(args.run, args.out)
    print(f"viewer -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

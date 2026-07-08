"""Loading of measured data: ROOT histogram files or phantom .npz directories.

Both sources are exposed through the same `Dataset` API so reconstruction and
evaluation run on real data and synthetic phantoms with zero code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hist2D:
    """A 2D counts histogram. values[i, j] counts bin (xedges[i:i+2], yedges[j:j+2])."""

    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    name: str = ""

    @property
    def xcenters(self) -> np.ndarray:
        return 0.5 * (self.xedges[:-1] + self.xedges[1:])

    @property
    def ycenters(self) -> np.ndarray:
        return 0.5 * (self.yedges[:-1] + self.yedges[1:])

    def crop(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> "Hist2D":
        """Restrict to bins fully inside xlim x ylim."""
        ix = np.where((self.xedges[:-1] >= xlim[0]) & (self.xedges[1:] <= xlim[1]))[0]
        iy = np.where((self.yedges[:-1] >= ylim[0]) & (self.yedges[1:] <= ylim[1]))[0]
        return Hist2D(
            values=self.values[np.ix_(ix, iy)],
            xedges=self.xedges[ix[0] : ix[-1] + 2],
            yedges=self.yedges[iy[0] : iy[-1] + 2],
            name=self.name,
        )

    def rebin(self, factor: int) -> "Hist2D":
        """Merge factor x factor blocks of bins (counts are summed)."""
        nx, ny = self.values.shape
        if nx % factor or ny % factor:
            raise ValueError(f"shape {self.values.shape} not divisible by {factor}")
        v = self.values.reshape(nx // factor, factor, ny // factor, factor).sum(axis=(1, 3))
        return Hist2D(
            values=v,
            xedges=self.xedges[::factor],
            yedges=self.yedges[::factor],
            name=self.name,
        )


def load_root_hist2d(path: str | Path, name: str) -> Hist2D:
    import uproot

    with uproot.open(path) as f:
        h = f[name]
        return Hist2D(
            values=h.values().astype(np.float64),
            xedges=h.axes[0].edges(),
            yedges=h.axes[1].edges(),
            name=name,
        )


def load_root_hist1d(path: str | Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, edges) for a 1D ROOT histogram."""
    import uproot

    with uproot.open(path) as f:
        h = f[name]
        return h.values().astype(np.float64), h.axes[0].edges()


@dataclass
class Dataset:
    """Measured counts for one campaign: a sky reference plus one file per position.

    hists maps source id ("sky", "pos0", "pos1", ...) -> {hist name -> Hist2D}.
    Histograms are loaded lazily for ROOT sources and eagerly for phantom dirs.
    """

    sources: dict  # id -> Path (ROOT file) or dict of Hist2D (phantom)
    meta: dict

    POSITION_IDS = ("pos0", "pos1")

    def hist(self, source: str, name: str) -> Hist2D:
        src = self.sources[source]
        if isinstance(src, dict):
            return src[name]
        return load_root_hist2d(src, name)

    def hist1d(self, source: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        src = self.sources[source]
        if isinstance(src, dict):
            raise KeyError(f"phantom datasets have no 1D histograms ({name})")
        return load_root_hist1d(src, name)

    @property
    def positions(self) -> list[str]:
        return [p for p in self.sources if p != "sky"]

    @property
    def is_phantom(self) -> bool:
        return self.meta.get("kind") == "phantom"


def load_dataset(spec: dict) -> Dataset:
    """Build a Dataset from a config `data` block.

    Real data:    {"sky": "data/...root", "pos0": "data/...root", "pos1": "data/...root"}
    Phantom dir:  {"phantom": "phantoms/p1"}
    """
    if "phantom" in spec:
        return load_phantom_dir(spec["phantom"])
    sources = {k: Path(v) for k, v in spec.items()}
    if "sky" not in sources:
        raise ValueError("data spec needs a 'sky' entry")
    return Dataset(sources=sources, meta={"kind": "root", "paths": {k: str(v) for k, v in spec.items()}})


def load_phantom_dir(path: str | Path) -> Dataset:
    """Load a phantom directory written by muontomo.phantom."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    sources: dict = {}
    for f in sorted(path.glob("counts_*.npz")):
        source_id = f.stem.removeprefix("counts_")
        with np.load(f) as z:
            sources[source_id] = {
                z["name"].item(): Hist2D(
                    values=z["values"], xedges=z["xedges"], yedges=z["yedges"], name=z["name"].item()
                )
            }
    meta["kind"] = "phantom"
    meta["dir"] = str(path)
    return Dataset(sources=sources, meta=meta)


def inspect_root_file(path: str | Path) -> dict:
    """Dump metadata (type, axes, integrals) for every histogram in a ROOT file."""
    import uproot

    out = {}
    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            obj = f[key]
            cls = getattr(obj, "classname", type(obj).__name__)
            entry: dict = {"class": cls}
            try:
                vals = obj.values()
                entry["sum"] = float(np.sum(vals))
                entry["max"] = float(np.max(vals))
                for i, ax in enumerate(obj.axes):
                    edges = ax.edges()
                    entry[f"axis{i}"] = {
                        "nbins": len(edges) - 1,
                        "lo": float(edges[0]),
                        "hi": float(edges[-1]),
                    }
            except Exception as exc:  # TProfile 'rate' has malformed axes in these files
                entry["error"] = f"{type(exc).__name__}: {exc}"
            out[key] = entry
    return out

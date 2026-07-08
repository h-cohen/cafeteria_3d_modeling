"""Playwright smoke test for a built viewer.html: no console errors, a
non-trivial mesh renders, camera presets produce non-blank screenshots.

    python -m muontomo.viewer.smoke --run runs/exp01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _preinstalled_chromium() -> str | None:
    """Locate the pre-installed Chromium binary (see environment's browsers dir)."""
    import glob
    import os

    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    hits = glob.glob(f"{root}/chromium-*/chrome-linux/chrome") + glob.glob(f"{root}/chromium/chrome-linux/chrome")
    return hits[0] if hits else None


def run_smoke(run_dir: str | Path) -> dict:
    from playwright.sync_api import sync_playwright

    run = Path(run_dir)
    viewer = run / "viewer.html"
    if not viewer.exists():
        raise FileNotFoundError(f"{viewer} not found -- run muontomo.viewer.build first")
    img_dir = run / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    report: dict = {"errors": [], "screenshots": []}

    with sync_playwright() as pw:
        launch_kwargs = {"args": ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"]}
        exe = _preinstalled_chromium()
        if exe:
            launch_kwargs["executable_path"] = exe
        try:
            browser = pw.chromium.launch(**launch_kwargs)
        except Exception:
            browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(f"file://{viewer.resolve()}")
        page.wait_for_function("window.__viewerReady === true", timeout=20000)

        state = page.evaluate("window.__viewerState()")
        report["initial_state"] = state
        assert state["triangles"] > 0, "marching cubes produced zero triangles"

        for preset in ["iso", "top", "front"]:
            page.evaluate(f"window.__frameCamera('{preset}')")
            page.wait_for_timeout(150)
            path = img_dir / f"viewer_{preset}.png"
            page.screenshot(path=str(path))
            report["screenshots"].append(str(path))
            _assert_non_blank(path)

        page.evaluate("window.__setState({threshold: 0.7})")
        page.wait_for_timeout(150)
        state2 = page.evaluate("window.__viewerState()")
        report["after_threshold_change"] = state2
        path = img_dir / "viewer_iso_hi.png"
        page.screenshot(path=str(path))
        report["screenshots"].append(str(path))
        _assert_non_blank(path)

        browser.close()

    report["errors"] = errors
    if errors:
        raise AssertionError(f"console errors during viewer smoke test: {errors}")
    return report


def _assert_non_blank(path: Path) -> None:
    import numpy as np
    from matplotlib import image as mpimg

    img = mpimg.imread(path)
    assert img[..., :3].std() > 0.01, f"{path} looks blank (std={img[..., :3].std():.4f})"
    n_unique = len(np.unique(img[..., :3].reshape(-1, 3), axis=0))
    assert n_unique > 100, f"{path} has only {n_unique} unique colors"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    report = run_smoke(args.run)
    print(f"OK: {report['initial_state']['triangles']:.0f} triangles, "
          f"{len(report['screenshots'])} screenshots, 0 console errors")


if __name__ == "__main__":
    main()

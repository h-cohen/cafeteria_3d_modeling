"""Smoke test for the per-run autofocus report (muontomo.focus_report):
synthetic inputs in, a valid 3-page PDF out."""

import json
import re

import numpy as np

from muontomo.config import RunConfig
from muontomo.focus_report import render_report


def test_render_report_writes_three_page_pdf(tmp_path):
    rng = np.random.default_rng(0)
    xs = np.arange(-5 + 0.05, 5, 0.1)
    ys = np.arange(-5 + 0.05, 5, 0.1)
    zs_scan = np.round(np.arange(6.0, 8.01, 0.1), 2)
    result = {
        "autofocus_z_m": 7.0,
        "quicklook_z_m": 7.4,
        "cv_scan": [{"z": float(z), "cv_mean": round(float((z - 7.0) ** 2 + 1.2), 4)}
                    for z in zs_scan],
        "height_map": {"median_z_m": 7.4, "iqr_z_m": 1.2},
        "solve_z_m": 7.0,
    }
    (tmp_path / "metrics.json").write_text(json.dumps({
        "beams": {"beams_x_data_m": [-3.12, -1.54, 0.14, 1.64, 3.32],
                  "n_beams_data": 5, "mean_abs_beam_offset_m": 0.10},
    }))
    ref_z = [6.0, 7.0, 8.0]
    ref_a = rng.random((3, xs.size, ys.size))
    ref_b = rng.random((3, xs.size, ys.size))

    pdf = render_report(tmp_path, RunConfig(), result, xs, ys, ref_z, ref_a, ref_b)
    assert pdf.exists() and pdf.stat().st_size > 10_000
    n_pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    assert n_pages == 3

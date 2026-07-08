"""Metric registry: every metric declares its direction so `compare` can judge deltas.

Metric roles (see README): phantom-truth metrics gate algorithm changes;
fidelity/cross-validation metrics gate calibration & geometry changes;
structural metrics are advisory (a strong prior can game them).
"""

from __future__ import annotations

REGISTRY: dict[str, dict] = {
    # fidelity
    "fidelity.chi2_ndof": {"higher_is_better": False, "band": 0.05},
    "fidelity.deviance_ndof": {"higher_is_better": False, "band": 0.05},
    "fidelity.chi2_aligned_gap": {"higher_is_better": False, "band": 0.05},
    # cross-position validation
    "crossval.cv_chi2": {"higher_is_better": False, "band": 0.05},
    "crossval.cv_pearson": {"higher_is_better": True, "band": 0.02},
    "crossval.cv_gap": {"higher_is_better": False, "band": 0.05},
    "crossval.binholdout_chi2": {"higher_is_better": False, "band": 0.05},
    # structure of the reconstructed ceiling slice
    "structure.volume_slice.periodicity_snr": {"higher_is_better": True, "band": 0.03},
    "structure.volume_slice.stripe_contrast": {"higher_is_better": True, "band": 0.03},
    "structure.volume_slice.flat_noise": {"higher_is_better": False, "band": 0.03},
    # volume plausibility
    "volume.neg_mass_frac": {"higher_is_better": False, "band": 0.01},
    "volume.z_eff_width_m": {"higher_is_better": False, "band": 0.03},
    # phantom ground truth (null on real data)
    "truth.rmse_scaled": {"higher_is_better": False, "band": 0.02},
    "truth.ssim3d": {"higher_is_better": True, "band": 0.02},
    "truth.iou_dense": {"higher_is_better": True, "band": 0.02},
    "truth.z_error_m": {"higher_is_better": False, "band": 0.02},
}


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a nested scorecard into dotted numeric keys."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and v is not None:
            out[key] = float(v)
    return out

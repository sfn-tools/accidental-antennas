"""Validate default params are within bounds for all models."""

from __future__ import annotations

import os

from sim import model_common as mc
from sim import model_dir24_quasiyagi
from sim import model_dir24_acs_monopole_yagi
from sim import model_dir24_acs_ursi_uniplanar
from sim import model_dir24_acs_ursi_ground
from sim import model_dir5_quasiyagi
from sim import model_dir5_ms_cps_dipole_yagi
from sim import model_omni_dual_ifa


def _check_model(model, constraints: mc.SliceConstraints) -> None:
    constraints = mc.apply_constraint_overrides(
        constraints, getattr(model, "DEFAULT_CONSTRAINTS", None)
    )
    defaults = model.default_params(constraints)
    bounds = model.param_bounds()
    for key, (low, high) in bounds.items():
        if key not in defaults:
            raise AssertionError(f"{model.MODEL_NAME} missing default for {key}")
        val = float(defaults[key])
        if not (low <= val <= high):
            raise AssertionError(
                f"{model.MODEL_NAME} default {key}={val} outside [{low}, {high}]"
            )


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    constraints = mc.load_constraints(os.path.join(root, "designs", "constraints.json"))
    models = [
        model_dir24_quasiyagi,
        model_dir24_acs_monopole_yagi,
        model_dir24_acs_ursi_uniplanar,
        model_dir24_acs_ursi_ground,
        model_dir5_quasiyagi,
        model_dir5_ms_cps_dipole_yagi,
        model_omni_dual_ifa,
    ]
    for model in models:
        _check_model(model, constraints)
    print("Default params are within bounds for all models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

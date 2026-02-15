"""5 GHz directional quasi-Yagi patch model."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc
from . import model_quasiyagi_patch as qp


MODEL_NAME = "dir5"
BANDS = {
    "5": {"f_low": 5.15e9, "f_high": 5.85e9, "f0": 5.50e9},
}


def default_params(constraints: mc.SliceConstraints) -> Dict:
    return {
        "scale": 0.61,
        "L_dir": 16.74,
        "W_dir": 1.45,
        "S_dir": 4.31,
        "L_dr": 16.74,
        "W_dr": 16.74,
        "S_ref": 6.23,
        "W_cps": 1.22,
        "S_cps": 0.20,
        "L_cps": 3.84,
        "L_taper": 7.67,
        "W_ref": 18.20,
        "L_ref": 15.35,
        "Lb1": 1.92,
        "Lb2": 1.92,
        "W_msl": 3.11,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "scale": (0.4, 2.5),
        "L_dir": (8.0, 24.0),
        "W_dir": (0.6, 3.0),
        "S_dir": (1.0, 10.0),
        "L_dr": (8.0, 20.0),
        "W_dr": (8.0, 24.0),
        "S_ref": (2.0, 12.0),
        "W_ref": (8.0, 24.0),
        "L_ref": (8.0, 20.0),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    return qp.build_geometry(params, constraints)

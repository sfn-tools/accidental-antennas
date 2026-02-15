"""2.4 GHz directional quasi-Yagi patch model."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc
from . import model_quasiyagi_patch as qp


MODEL_NAME = "dir24"
BANDS = {
    "24": {"f_low": 2.40e9, "f_high": 2.48e9, "f0": 2.44e9},
}


def default_params(constraints: mc.SliceConstraints) -> Dict:
    return {
        "scale": 1.72,
        "L_dir": 37.74,
        "W_dir": 3.20,
        "S_dir": 9.71,
        "L_dr": 37.74,
        "W_dr": 37.74,
        "S_ref": 14.04,
        "W_cps": 2.75,
        "S_cps": 0.45,
        "L_cps": 8.66,
        "L_taper": 17.28,
        "W_ref": 41.00,
        "L_ref": 34.60,
        "Lb1": 4.33,
        "Lb2": 4.33,
        "W_msl": 3.11,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "scale": (0.4, 2.5),
        "L_dir": (16.0, 50.0),
        "W_dir": (0.8, 4.0),
        "S_dir": (4.0, 16.0),
        "L_dr": (16.0, 50.0),
        "W_dr": (10.0, 50.0),
        "S_ref": (6.0, 20.0),
        "W_ref": (12.0, 50.0),
        "L_ref": (12.0, 45.0),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    return qp.build_geometry(params, constraints)

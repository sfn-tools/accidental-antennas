"""URSI 2.4 GHz ACS-fed compact Yagi-Uda (uniplanar)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from . import model_common as mc


MODEL_NAME = "dir24_acs_ursi_uni"
BANDS = {
    "24": {"f_low": 2.40e9, "f_high": 2.48e9, "f0": 2.44e9},
}

DEFAULT_SUBSTRATE = {
    "eps_r": 4.4,
    "tan_delta": 0.02,
    "thickness_mm": 1.6,
    "copper_thickness_mm": 0.035,
}

DEFAULT_CONSTRAINTS = {
    "slice_angle_deg": 80.0,
}


def default_params(constraints: mc.SliceConstraints) -> Dict:
    return {
        "scale": 1.0,
        "L": 47.0,
        "W": 24.0,
        "lr": 46.0,
        "wr": 4.0,
        "lp": 45.0,
        "wp": 4.0,
        "lg": 22.0,
        "wg": 4.0,
        "g": 1.0,
        "wd": 9.0,
        "port_offset": 1.0,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "scale": (0.6, 2.0),
        "L": (30.0, 55.0),
        "W": (16.0, 28.0),
        "lr": (30.0, 52.0),
        "wr": (2.0, 6.0),
        "lp": (30.0, 52.0),
        "wp": (2.0, 6.0),
        "lg": (12.0, 30.0),
        "wg": (2.0, 6.0),
        "g": (0.5, 2.0),
        "wd": (5.0, 15.0),
        "port_offset": (0.5, 3.0),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    scale = float(params.get("scale", 1.0))
    L = float(params["L"]) * scale
    W = float(params["W"]) * scale
    lr = float(params["lr"]) * scale
    wr = float(params["wr"]) * scale
    lp = float(params["lp"]) * scale
    wp = float(params["wp"]) * scale
    lg = float(params["lg"]) * scale
    wg = float(params["wg"]) * scale
    g = float(params["g"]) * scale
    wd = float(params["wd"]) * scale
    port_offset = float(params.get("port_offset", 1.0)) * scale

    y_bot = -L / 2.0
    outer = constraints.outer_radius_mm - constraints.keepout_edge_mm
    width_total = wr + wd + wp + g + wg
    y_edge = abs(y_bot)
    if y_edge >= outer:
        x_right = 0.0
    else:
        x_right = min(outer, math.sqrt(max(outer * outer - y_edge * y_edge, 0.0)))
    half_rad = math.radians(constraints.slice_angle_deg / 2.0)
    x0_min = abs(y_bot) / math.tan(half_rad) if half_rad > 0 else 0.0
    x0_max = x_right - width_total
    margin_req = max(0.0, W - width_total)
    margin_allow = max(0.0, x0_max - x0_min)
    x0 = x0_max - min(margin_req, margin_allow)

    x_ref0, x_ref1 = x0, x0 + wr
    x_m0, x_m1 = x_ref1 + wd, x_ref1 + wd + wp
    x_g0, x_g1 = x_m1 + g, x_m1 + g + wg

    geom = mc.Geometry()
    geom.feed_point = (x_m1 + g / 2.0, y_bot + max(0.5, min(port_offset, lp - 0.5)))
    geom.ground_polys = []

    geom.top_polys.extend(
        [
            mc.rect_polygon("acs_reflector", x_ref0, x_ref1, y_bot, y_bot + lr),
            mc.rect_polygon("acs_monopole", x_m0, x_m1, y_bot, y_bot + lp),
            mc.rect_polygon("acs_ground", x_g0, x_g1, y_bot, y_bot + lg),
        ]
    )

    port_w = max(0.5, min(1.5, min(wp, wg)))
    y_port = y_bot + max(0.5, min(port_offset, min(lp, lg) - 0.5))
    geom.meta["port_defs"] = [
        {
            "start": [x_m1, y_port - port_w / 2.0, 0.0],
            "stop": [x_g0, y_port + port_w / 2.0, None],
            "exc_dir": "x",
            "R": 50,
        }
    ]

    constraint_info = {
        "min_trace_mm": min(wr, wp, wg),
        "min_gap_mm": g,
    }

    return geom, constraint_info

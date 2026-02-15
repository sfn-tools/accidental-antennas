"""Dual-band omni IFA/PIFA-style planar model."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc


MODEL_NAME = "omni"
BANDS = {
    "24": {"f_low": 2.40e9, "f_high": 2.48e9, "f0": 2.44e9},
    "5": {"f_low": 5.15e9, "f_high": 5.85e9, "f0": 5.50e9},
}


def default_params(constraints: mc.SliceConstraints) -> Dict:
    return {
        "feed_offset_mm": 4.0,
        "feed_line_len_mm": 26.0,
        "feed_line_w_mm": 1.0,
        "arm_w_mm": 1.6,
        "arm_len_24_mm": 24.0,
        "arm_len_5_mm": 10.0,
        "arm_sep_mm": 6.0,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "feed_offset_mm": (1.0, 12.0),
        "feed_line_len_mm": (10.0, 60.0),
        "feed_line_w_mm": (0.2, 3.0),
        "arm_w_mm": (0.6, 4.0),
        "arm_len_24_mm": (14.0, 40.0),
        "arm_len_5_mm": (4.0, 20.0),
        "arm_sep_mm": (2.0, 16.0),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    feed_x = constraints.inner_radius_mm + params["feed_offset_mm"]
    feed_w = params["feed_line_w_mm"]
    feed_len = params["feed_line_len_mm"]

    arm_w = params["arm_w_mm"]
    arm_len_24 = params["arm_len_24_mm"]
    arm_len_5 = params["arm_len_5_mm"]
    arm_sep = params["arm_sep_mm"]

    arm_x = feed_x + feed_len
    arm2_x = arm_x + arm_sep
    feed_end = arm2_x + arm_w / 2.0

    geom = mc.Geometry()
    geom.feed_point = (feed_x, 0.0)
    geom.ground_polys = [
        mc.wedge_polygon(
            constraints.inner_radius_mm,
            constraints.outer_radius_mm,
            constraints.slice_angle_deg,
        )
    ]

    geom.top_polys.append(
        mc.rect_polygon(
            "feed",
            feed_x,
            feed_end,
            -feed_w / 2.0,
            feed_w / 2.0,
        )
    )

    geom.top_polys.append(
        mc.rect_polygon(
            "arm_24",
            arm_x - arm_w / 2.0,
            arm_x + arm_w / 2.0,
            -arm_len_24 / 2.0,
            arm_len_24 / 2.0,
        )
    )

    geom.top_polys.append(
        mc.rect_polygon(
            "arm_5",
            arm2_x - arm_w / 2.0,
            arm2_x + arm_w / 2.0,
            -arm_len_5 / 2.0,
            arm_len_5 / 2.0,
        )
    )

    gap = arm_sep - arm_w
    constraint_info = {
        "min_trace_mm": min(feed_w, arm_w),
        "min_gap_mm": gap if gap > 0 else 0.0,
    }

    return geom, constraint_info

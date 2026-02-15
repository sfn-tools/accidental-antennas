"""Shared geometry for patch-driven quasi-Yagi models."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc


W_B2_RATIO = 0.737
W_B3_RATIO = 0.558


def derived_widths(w_msl: float) -> Dict[str, float]:
    return {
        "Wb1": w_msl,
        "Wb2": w_msl * W_B2_RATIO,
        "Wb3": w_msl * W_B3_RATIO,
    }


def _taper_poly(
    name: str,
    x0: float,
    x1: float,
    y0_low: float,
    y0_high: float,
    y1_low: float,
    y1_high: float,
) -> mc.Polygon2D:
    return mc.Polygon2D(
        name=name,
        points=[(x0, y0_low), (x0, y0_high), (x1, y1_high), (x1, y1_low)],
    )


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    scale = float(params.get("scale", 1.0))

    def _sx(name: str) -> float:
        return float(params[name]) * scale

    def _sy(name: str) -> float:
        return float(params[name])

    L_dir = _sx("L_dir")
    W_dir = _sx("W_dir")
    S_dir = _sx("S_dir")

    L_dr = _sx("L_dr")
    W_dr = _sy("W_dr")

    S_ref = _sx("S_ref")

    W_cps = _sy("W_cps")
    S_cps = _sy("S_cps")
    L_cps = _sx("L_cps")

    L_taper = _sx("L_taper")

    W_ref = _sy("W_ref")
    L_ref = _sx("L_ref")

    Lb1 = _sx("Lb1")
    Lb2 = _sx("Lb2")
    W_msl = float(params["W_msl"])

    widths = derived_widths(W_msl)
    Wb1 = widths["Wb1"]
    Wb2 = widths["Wb2"]
    Wb3 = widths["Wb3"]

    x_feed = constraints.feed_location_mm
    x_seg1 = x_feed + Lb1
    x_seg2 = x_seg1 + Lb2
    x_taper = x_seg2 + L_taper
    x_cps = x_taper + L_cps
    x_driver_end = x_cps + W_dr

    geom = mc.Geometry()
    geom.feed_point = (x_feed, 0.0)

    geom.top_polys.append(mc.rect_polygon("feed_wb1", x_feed, x_seg1, -Wb1 / 2.0, Wb1 / 2.0))
    geom.top_polys.append(mc.rect_polygon("feed_wb2", x_seg1, x_seg2, -Wb2 / 2.0, Wb2 / 2.0))

    geom.top_polys.append(
        _taper_poly(
            "taper_upper",
            x_seg2,
            x_taper,
            0.0,
            Wb3 / 2.0,
            S_cps / 2.0,
            S_cps / 2.0 + W_cps,
        )
    )
    geom.top_polys.append(
        _taper_poly(
            "taper_lower",
            x_seg2,
            x_taper,
            -Wb3 / 2.0,
            0.0,
            -S_cps / 2.0 - W_cps,
            -S_cps / 2.0,
        )
    )

    geom.top_polys.append(
        mc.rect_polygon(
            "cps_upper",
            x_taper,
            x_cps,
            S_cps / 2.0,
            S_cps / 2.0 + W_cps,
        )
    )
    geom.top_polys.append(
        mc.rect_polygon(
            "cps_lower",
            x_taper,
            x_cps,
            -S_cps / 2.0 - W_cps,
            -S_cps / 2.0,
        )
    )

    geom.top_polys.append(
        mc.rect_polygon(
            "driver_upper",
            x_cps,
            x_driver_end,
            S_cps / 2.0,
            S_cps / 2.0 + L_dr,
        )
    )
    geom.top_polys.append(
        mc.rect_polygon(
            "driver_lower",
            x_cps,
            x_driver_end,
            -S_cps / 2.0 - L_dr,
            -S_cps / 2.0,
        )
    )

    x_dir_start = x_driver_end + S_dir
    geom.top_polys.append(
        mc.rect_polygon(
            "director",
            x_dir_start,
            x_dir_start + W_dir,
            -L_dir / 2.0,
            L_dir / 2.0,
        )
    )

    x_ref_stop = x_cps - S_ref
    x_ref_start = x_ref_stop - L_ref
    min_x = constraints.inner_radius_mm + constraints.keepout_edge_mm
    if x_ref_start < min_x:
        x_ref_start = min_x
    if x_feed < x_ref_start:
        x_ref_start = max(min_x, x_feed - 0.5)
    if x_feed > x_ref_stop:
        x_ref_stop = x_feed + 0.5
    if x_ref_stop <= x_ref_start:
        x_ref_stop = x_ref_start + 0.5

    y_ref = W_ref / 2.0
    geom.ground_polys = [mc.rect_polygon("ground", x_ref_start, x_ref_stop, -y_ref, y_ref)]

    constraint_info = {
        "min_trace_mm": min(Wb1, Wb2, Wb3, W_cps, W_dr),
        "min_gap_mm": min(S_cps, S_dir, S_ref),
    }

    geom.meta["derived_widths"] = widths
    geom.meta["scale"] = scale

    return geom, constraint_info

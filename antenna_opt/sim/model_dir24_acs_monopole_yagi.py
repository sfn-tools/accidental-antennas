"""ACS-fed monopole-driven compact Yagi-Uda at 2.4 GHz."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc


MODEL_NAME = "dir24_acs"
BANDS = {
    "24": {"f_low": 2.40e9, "f_high": 2.48e9, "f0": 2.44e9},
}

DEFAULT_SUBSTRATE = {
    "eps_r": 4.4,
    "tan_delta": 0.02,
    "thickness_mm": 1.6,
    "copper_thickness_mm": 0.035,
}


def default_params(constraints: mc.SliceConstraints) -> Dict:
    return {
        "scale": 1.0,
        "L": 2.0,
        "W": 22.0,
        "lr": 32.0,
        "wg": 3.0,
        "wp": 3.0,
        "wr": 3.0,
        "wd": 3.0,
        "g": 0.6,
        "lg": 45.0,
        "lp": 32.0,
        "S_ref": 2.0,
        "L_hat": 4.0,
        "W_hat": 2.0,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "scale": (0.7, 1.2),
        "L": (1.0, 6.0),
        "W": (16.0, 28.0),
        "lr": (22.0, 36.0),
        "wg": (2.0, 5.0),
        "wp": (2.0, 5.0),
        "wr": (2.0, 6.0),
        "wd": (2.0, 6.0),
        "g": (0.4, 1.5),
        "lg": (35.0, 55.0),
        "lp": (24.0, 38.0),
        "S_ref": (1.0, 6.0),
        "L_hat": (0.0, 8.0),
        "W_hat": (0.8, 3.5),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    scale = float(params.get("scale", 1.0))
    L = float(params["L"])
    W = float(params["W"]) * scale
    lr = float(params["lr"]) * scale
    lg = float(params["lg"])
    lp = float(params["lp"]) * scale
    g = float(params["g"])
    wg = float(params["wg"])
    wp = float(params["wp"])
    wr = float(params["wr"])
    wd = float(params["wd"])
    s_ref = float(params["S_ref"])
    L_hat = float(params["L_hat"])
    W_hat = float(params["W_hat"])

    x_feed = constraints.feed_location_mm
    x_line_end = x_feed + lg

    geom = mc.Geometry()
    geom.feed_point = (x_feed, 0.0)
    geom.ground_polys = [
        mc.wedge_polygon(
            constraints.inner_radius_mm + constraints.keepout_edge_mm,
            constraints.outer_radius_mm - constraints.keepout_edge_mm,
            constraints.slice_angle_deg,
        )
    ]

    # ACS feed: signal strip + asymmetric ground strip on top copper.
    geom.top_polys.append(mc.rect_polygon("feed", x_feed, x_line_end, -wp / 2.0, wp / 2.0))
    geom.top_polys.append(
        mc.rect_polygon(
            "acs_ground",
            x_feed,
            x_line_end,
            wp / 2.0 + g,
            wp / 2.0 + g + wg,
        )
    )
    x_port = x_feed + max(0.2, min(1.0, 0.25 * lg))
    if x_port > x_line_end - 0.2:
        x_port = x_line_end - 0.2
    if x_port <= x_feed:
        x_port = x_feed + 0.2
    port_w = max(0.3, min(1.0, wp))
    geom.meta["port_defs"] = [
        {
            "start": [x_port - port_w / 2.0, 0.0, 0.0],
            "stop": [x_port + port_w / 2.0, wp / 2.0 + g + wg / 2.0, None],
            "exc_dir": "y",
            "R": 50,
        }
    ]

    # Reflector behind the driver (edge-to-edge gap = s_ref).
    x_ref1 = x_line_end - s_ref
    x_ref0 = x_ref1 - wr
    geom.top_polys.append(mc.rect_polygon("reflector", x_ref0, x_ref1, -lr / 2.0, lr / 2.0))

    # Monopole driver at the end of the feed line.
    x_drv0 = x_line_end
    x_drv1 = x_drv0 + wp
    geom.top_polys.append(mc.rect_polygon("driver", x_drv0, x_drv1, -lp / 2.0, lp / 2.0))
    if L_hat > 0 and W_hat > 0:
        geom.top_polys.append(
            mc.rect_polygon(
                "driver_hat_top",
                x_drv0 - L_hat,
                x_drv0,
                lp / 2.0 - W_hat,
                lp / 2.0,
            )
        )
        geom.top_polys.append(
            mc.rect_polygon(
                "driver_hat_bot",
                x_drv0 - L_hat,
                x_drv0,
                -lp / 2.0,
                -lp / 2.0 + W_hat,
            )
        )

    # Director forward of the driver (gap = L, length = W, width = wd).
    x_dir0 = x_drv1 + L
    x_dir1 = x_dir0 + wd
    geom.top_polys.append(mc.rect_polygon("director", x_dir0, x_dir1, -W / 2.0, W / 2.0))

    min_trace_candidates = [wg, wp, wr, wd, W_hat]
    min_trace = min(val for val in min_trace_candidates if val > 0)
    min_gap = min(g, s_ref, L)

    constraint_info = {
        "min_trace_mm": min_trace,
        "min_gap_mm": min_gap,
    }

    return geom, constraint_info

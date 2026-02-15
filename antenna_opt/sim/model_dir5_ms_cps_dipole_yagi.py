"""5 GHz printed Yagi-Uda with microstrip-to-CPS transition."""

from __future__ import annotations

from typing import Dict, Tuple

from . import model_common as mc


MODEL_NAME = "dir5_ms_cps"
BANDS = {
    "5": {"f_low": 5.15e9, "f_high": 5.85e9, "f0": 5.50e9},
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
        "L_drv": 17.0,
        "L_dir": 11.5,
        "L_ref": 21.5,
        "W_elem": 2.0,
        "d_dir": 5.5,
        "d_ref": 4.5,
        "lg": 6.2,
        "wg": 9.0,
        "h1": 2.6,
        "W_msl": 3.11,
        "W_cps": 1.0,
        "S_cps": 0.4,
        "L_taper": 2.0,
        "slot_l": 2.0,
        "slot_w": 1.0,
        "L_hat": 0.0,
        "W_hat": 0.0,
        "fold_gap": 0.0,
        "fold_w": 0.0,
        "x_offset": 20.0,
    }


def param_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "scale": (0.4, 2.5),
        "L_drv": (8.0, 22.0),
        "L_dir": (4.0, 16.0),
        "L_ref": (8.0, 26.0),
        "W_elem": (1.0, 3.5),
        "d_dir": (2.0, 8.0),
        "d_ref": (2.0, 8.0),
        "lg": (4.0, 12.0),
        "wg": (6.0, 14.0),
        "h1": (1.0, 6.0),
        "W_cps": (0.6, 2.5),
        "S_cps": (0.2, 1.5),
        "L_taper": (0.5, 6.0),
        "slot_l": (0.0, 6.0),
        "slot_w": (0.0, 3.0),
        "L_hat": (0.0, 6.0),
        "W_hat": (0.0, 3.0),
        "fold_gap": (0.0, 1.5),
        "fold_w": (0.0, 2.5),
        "x_offset": (10.0, 30.0),
    }


def build_geometry(params: Dict, constraints: mc.SliceConstraints) -> Tuple[mc.Geometry, Dict]:
    scale = float(params.get("scale", 1.0))
    L_drv = float(params["L_drv"]) * scale
    L_dir = float(params["L_dir"]) * scale
    L_ref = float(params["L_ref"]) * scale
    d_dir = float(params["d_dir"]) * scale
    d_ref = float(params["d_ref"]) * scale
    lg = float(params["lg"]) * scale
    wg = float(params["wg"])
    h1 = float(params["h1"]) * scale
    L_taper = float(params["L_taper"])
    slot_l = float(params["slot_l"])

    W_elem = float(params["W_elem"])
    W_msl = float(params["W_msl"])
    W_cps = float(params["W_cps"])
    S_cps = float(params["S_cps"])
    slot_w = float(params["slot_w"])
    L_hat = float(params.get("L_hat", 0.0))
    W_hat = float(params.get("W_hat", 0.0))
    fold_gap = float(params.get("fold_gap", 0.0))
    fold_w = float(params.get("fold_w", 0.0))

    x_offset = float(params.get("x_offset", 0.0))
    x_feed = constraints.feed_location_mm + x_offset
    x_ground_start = x_feed
    x_ground_end = x_feed + lg
    x_cps_start = x_ground_end
    x_driver_center = x_cps_start + h1
    x_ref_center = x_driver_center - d_ref
    x_dir_center = x_driver_center + d_dir

    geom = mc.Geometry()
    geom.feed_point = (x_feed, 0.0)

    # Microstrip feed line on top copper.
    geom.top_polys.append(
        mc.rect_polygon("msl", x_feed, x_ground_end, -W_msl / 2.0, W_msl / 2.0)
    )

    # Step-taper into the CPS width.
    if L_taper > 0:
        x_taper_start = x_ground_end - L_taper
        W_taper = max(W_msl, 2.0 * W_cps + S_cps)
        geom.top_polys.append(
            mc.rect_polygon(
                "msl_taper",
                x_taper_start,
                x_ground_end,
                -W_taper / 2.0,
                W_taper / 2.0,
            )
        )

    # CPS feed lines.
    y_gap = S_cps / 2.0
    x_cps_end = x_driver_center
    geom.top_polys.append(
        mc.rect_polygon("cps_upper", x_cps_start, x_cps_end, y_gap, y_gap + W_cps)
    )
    geom.top_polys.append(
        mc.rect_polygon("cps_lower", x_cps_start, x_cps_end, -y_gap - W_cps, -y_gap)
    )

    x_port = x_feed + max(0.2, min(1.0, 0.2 * lg))
    if x_port > x_ground_end - 0.2:
        x_port = x_ground_end - 0.2
    geom.meta["port_defs"] = [
        {
            "start": [x_port, 0.0, 0.0],
            "stop": [x_port, 0.0, None],
            "exc_dir": "z",
            "R": 50,
        }
    ]

    # Driver dipole arms (split around the feed gap).
    x_drv0 = x_driver_center - W_elem / 2.0
    x_drv1 = x_driver_center + W_elem / 2.0
    geom.top_polys.append(
        mc.rect_polygon("driver_upper", x_drv0, x_drv1, y_gap, L_drv / 2.0)
    )
    geom.top_polys.append(
        mc.rect_polygon("driver_lower", x_drv0, x_drv1, -L_drv / 2.0, -y_gap)
    )
    if L_hat > 0 and W_hat > 0:
        y_hat = max(y_gap, min(L_drv / 2.0 - W_hat, 0.3 * L_drv))
        geom.top_polys.append(
            mc.rect_polygon("driver_hat_top", x_drv0 - L_hat, x_drv0, y_hat, y_hat + W_hat)
        )
        geom.top_polys.append(
            mc.rect_polygon(
                "driver_hat_bot", x_drv0 - L_hat, x_drv0, -y_hat - W_hat, -y_hat
            )
        )
    if fold_gap > 0 and fold_w > 0:
        x_fold0 = x_drv0 - fold_gap
        x_fold1 = x_fold0 - fold_w
        geom.top_polys.append(
            mc.rect_polygon("driver_fold", x_fold1, x_fold0, -L_drv / 2.0, L_drv / 2.0)
        )
        geom.top_polys.append(
            mc.rect_polygon(
                "driver_fold_top",
                x_fold1,
                x_drv0,
                L_drv / 2.0 - W_elem,
                L_drv / 2.0,
            )
        )
        geom.top_polys.append(
            mc.rect_polygon(
                "driver_fold_bot",
                x_fold1,
                x_drv0,
                -L_drv / 2.0,
                -L_drv / 2.0 + W_elem,
            )
        )

    # Director (single strip).
    x_dir0 = x_dir_center - W_elem / 2.0
    x_dir1 = x_dir_center + W_elem / 2.0
    geom.top_polys.append(mc.rect_polygon("director", x_dir0, x_dir1, -L_dir / 2.0, L_dir / 2.0))

    # Reflector (split arms with the same feed gap).
    x_ref0 = x_ref_center - W_elem / 2.0
    x_ref1 = x_ref_center + W_elem / 2.0
    arm_ref = L_ref / 2.0 - y_gap
    if arm_ref > 0:
        geom.top_polys.append(
            mc.rect_polygon("ref_upper", x_ref0, x_ref1, y_gap, y_gap + arm_ref)
        )
        geom.top_polys.append(
            mc.rect_polygon("ref_lower", x_ref0, x_ref1, -y_gap - arm_ref, -y_gap)
        )

    # Finite ground plane with optional slot near the transition.
    if slot_l > 0 and slot_w > 0:
        x_slot_start = x_ground_end - slot_l
        if x_slot_start > x_ground_start:
            geom.ground_polys.append(
                mc.rect_polygon("gnd", x_ground_start, x_slot_start, -wg / 2.0, wg / 2.0)
            )
        geom.ground_polys.append(
            mc.rect_polygon(
                "gnd_upper",
                x_slot_start,
                x_ground_end,
                slot_w / 2.0,
                wg / 2.0,
            )
        )
        geom.ground_polys.append(
            mc.rect_polygon(
                "gnd_lower",
                x_slot_start,
                x_ground_end,
                -wg / 2.0,
                -slot_w / 2.0,
            )
        )
    else:
        geom.ground_polys.append(
            mc.rect_polygon("gnd", x_ground_start, x_ground_end, -wg / 2.0, wg / 2.0)
        )

    min_trace_candidates = [W_elem, W_msl, W_cps]
    if W_hat > 0:
        min_trace_candidates.append(W_hat)
    if fold_w > 0:
        min_trace_candidates.append(fold_w)
    min_trace = min(min_trace_candidates)
    min_gap_candidates = [S_cps]
    if slot_w > 0:
        min_gap_candidates.append(slot_w)
    if fold_gap > 0:
        min_gap_candidates.append(fold_gap)
    min_gap = min(min_gap_candidates)

    constraint_info = {
        "min_trace_mm": min_trace,
        "min_gap_mm": min_gap,
    }

    return geom, constraint_info

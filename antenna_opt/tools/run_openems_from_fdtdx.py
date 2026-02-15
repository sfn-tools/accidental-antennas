"""Run an openEMS simulation from an FDTDX run geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np

from sim import matching
from sim import model_common as mc
from sim import post
from tools import visualize_pattern


def _load_fdtdx_run(run_dir: str) -> Tuple[Dict, np.ndarray, float]:
    params_path = os.path.join(run_dir, "params.json")
    geom_path = os.path.join(run_dir, "geometry.npz")
    if not os.path.exists(params_path) or not os.path.exists(geom_path):
        raise FileNotFoundError("Missing params.json or geometry.npz in FDTDX run directory")
    with open(params_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)
    data = np.load(geom_path)
    design_indices = data["design_indices"]
    resolution_mm = float(data["resolution_mm"])
    return params, design_indices, resolution_mm


def _wedge_points(inner_r: float, outer_r: float, angle_deg: float, n: int = 48) -> List[Tuple[float, float]]:
    half = math.radians(angle_deg / 2.0)
    angles_outer = [(-half) + i * (2 * half) / (n - 1) for i in range(n)]
    angles_inner = list(reversed(angles_outer))
    outer = [(outer_r * math.cos(a), outer_r * math.sin(a)) for a in angles_outer]
    inner = [(inner_r * math.cos(a), inner_r * math.sin(a)) for a in angles_inner]
    return outer + inner


def _normalize(points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, min_y = min(xs), min(ys)
    shifted = [(x - min_x, y - min_y) for x, y in points]
    return shifted, (min_x, min_y)


def _rects_from_mask(mask: np.ndarray, resolution_mm: float, origin_x: float, origin_y: float):
    rects = []
    nx, ny = mask.shape[:2]
    for ix in range(nx):
        row = mask[ix, :]
        run_start = None
        for iy in range(ny + 1):
            val = row[iy] if iy < ny else 0
            if val and run_start is None:
                run_start = iy
            elif run_start is not None and (not val or iy == ny):
                x0 = origin_x + ix * resolution_mm
                y0 = origin_y + run_start * resolution_mm
                width = resolution_mm
                height = (iy - run_start) * resolution_mm
                rects.append((x0, y0, x0 + width, y0 + height))
                run_start = None
    return rects


def _svg_points(points, min_x, max_y, scale, margin):
    coords = []
    for x, y in points:
        sx = (x - min_x) * scale + margin
        sy = (max_y - y) * scale + margin
        coords.append(f"{sx:.3f},{sy:.3f}")
    return " ".join(coords)


def _write_svg(path: str, polys: List[mc.Polygon2D]) -> None:
    if not polys:
        return
    all_pts = [pt for poly in polys for pt in poly.points]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale = 5.0
    margin = 10.0
    width = (max_x - min_x) * scale + 2 * margin
    height = (max_y - min_y) * scale + 2 * margin

    lines = [
        "<svg xmlns=\"http://www.w3.org/2000/svg\"",
        f"     width=\"{width:.1f}px\" height=\"{height:.1f}px\"",
        f"     viewBox=\"0 0 {width:.1f} {height:.1f}\">",
        "  <g fill=\"none\" stroke=\"#000\" stroke-width=\"1\">",
    ]
    for poly in polys:
        pts = _svg_points(poly.points, min_x, max_y, scale, margin)
        lines.append(f"    <polygon points=\"{pts}\" />")
    lines.append("  </g>")
    lines.append("</svg>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _export_geometry(out_dir: str, geom: mc.Geometry) -> None:
    _write_svg(os.path.join(out_dir, "copper_top.svg"), geom.top_polys)
    if geom.ground_polys:
        _write_svg(os.path.join(out_dir, "copper_ground.svg"), geom.ground_polys)


def _prune_run(out_dir: str) -> None:
    keep = {
        "copper_top.svg",
        "copper_ground.svg",
        "fdtdx_meta.json",
        "meta.json",
        "metrics.json",
        "params.json",
        "pattern_3d.vtk",
        "s11.csv",
        "s11.png",
        "s11_matched.csv",
        "s11_matched.png",
        "s11_matched_smith.csv",
        "s11_matched3.csv",
        "s11_matched3.png",
        "s11_matched3_smith.csv",
    }
    for root, dirs, files in os.walk(out_dir, topdown=False):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), out_dir)
            if rel in keep:
                continue
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            path = os.path.join(root, name)
            try:
                if not os.listdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


def _fdtdx_port_offsets_mm(cfg: Dict, resolution_mm: float) -> Tuple[float, float, float, float]:
    feed_in_len = float(cfg["feed_length_mm"]) - float(cfg["feed_stub_length_mm"])
    if feed_in_len <= 0:
        raise ValueError("feed_length_mm must exceed feed_stub_length_mm")
    source_gap = float(cfg["source_gap_mm"])
    pad = 3.0 * float(resolution_mm)
    port_i_len = source_gap + 2.0 * pad
    port_det_len = max(source_gap, port_i_len, float(resolution_mm))

    max_source_offset = max(0.0, feed_in_len - (source_gap + port_det_len))
    max_port_offset = max(0.0, feed_in_len - port_det_len)
    source_offset = max(float(cfg.get("source_offset_mm", 0.0)), 0.0)
    port_offset = max(float(cfg.get("port_offset_mm", 0.0)), 0.0)
    if source_offset > max_source_offset:
        source_offset = max_source_offset
    if port_offset > max_port_offset:
        port_offset = max_port_offset
    min_port_offset = source_offset + source_gap
    if port_offset < min_port_offset:
        port_offset = min_port_offset
    if port_offset > max_port_offset:
        port_offset = max_port_offset
    return source_offset, port_offset, source_gap, port_det_len


def _build_geometry(
    cfg: Dict,
    design_indices: np.ndarray,
    threshold: float,
    port_mode: str,
    port_type: str,
    gap_height_mm: float,
    clip_to_wedge: bool,
    overlap_mm: float,
) -> mc.Geometry:
    inner_r = float(cfg["inner_radius_mm"])
    outer_r = float(cfg["outer_radius_mm"])
    angle = float(cfg["slice_angle_deg"])
    wedge_pts = _wedge_points(inner_r, outer_r, angle)
    shifted, offset = _normalize(wedge_pts)
    min_x, max_x, min_y, max_y = mc.geometry_bounds(
        mc.Geometry(top_polys=[], ground_polys=[mc.Polygon2D("w", shifted)])
    )
    wedge_w = max_x - min_x
    wedge_h = max_y - min_y

    geom = mc.Geometry()
    ground_trim = float(cfg.get("ground_trim_mm", 0.0))
    min_gap = float(cfg.get("min_gap_mm", 0.2))
    if ground_trim > 0:
        min_outer = inner_r + max(min_gap, 0.5)
        ground_outer_r = max(min_outer, outer_r - ground_trim)
    else:
        ground_outer_r = outer_r
    ground_pts = _wedge_points(inner_r, ground_outer_r, angle)
    ground_shifted = [(x - offset[0], y - offset[1]) for x, y in ground_pts]
    geom.ground_polys.append(mc.Polygon2D(name="ground", points=ground_shifted))

    design_mask = (design_indices[..., 0] >= threshold).astype(np.uint8)
    resolution_mm = float(cfg["base_resolution_mm"]) * float(cfg.get("resolution_scale", 1.0))
    resolution_mm = float(cfg.get("resolution_mm", resolution_mm))

    device_origin_x = float(cfg["feed_offset_mm"]) + float(cfg["feed_length_mm"])
    device_origin_y = (wedge_h - float(cfg["design_width_mm"])) * 0.5
    rects = _rects_from_mask(design_mask, resolution_mm, device_origin_x, device_origin_y)
    if clip_to_wedge:
        half_angle = angle * 0.5
        clipped = []
        for x0, y0, x1, y1 in rects:
            cx = 0.5 * (x0 + x1) + offset[0]
            cy = 0.5 * (y0 + y1) + offset[1]
            r = math.hypot(cx, cy)
            ang = math.degrees(math.atan2(cy, cx))
            if r < inner_r or r > outer_r or abs(ang) > half_angle:
                continue
            clipped.append((x0, y0, x1, y1))
        rects = clipped
    for idx, (x0, y0, x1, y1) in enumerate(rects):
        geom.top_polys.append(mc.rect_polygon(f"pix_{idx}", x0, x1, y0, y1))

    feed_width = float(cfg["feed_width_mm"])
    port_width = float(cfg.get("port_width_mm", feed_width))
    feed_offset = float(cfg["feed_offset_mm"])
    feed_length = float(cfg["feed_length_mm"])
    feed_stub = float(cfg["feed_stub_length_mm"])
    source_gap = float(cfg["source_gap_mm"])
    if overlap_mm < 0:
        overlap_mm = 0.0

    y_center = wedge_h * 0.5
    y0 = y_center - feed_width * 0.5
    y1 = y_center + feed_width * 0.5

    port_offsets = None
    if port_mode in ("fdtdx", "fdtdx_line"):
        port_offsets = _fdtdx_port_offsets_mm(cfg, resolution_mm)

    if port_mode == "gap":
        feed_in_len = feed_length - feed_stub - source_gap
        if feed_in_len > 0:
            geom.top_polys.append(mc.rect_polygon("feed_line", feed_offset, feed_offset + feed_in_len, y0, y1))
        stub_x0 = feed_offset + feed_length - feed_stub
        stub_x1 = feed_offset + feed_length + overlap_mm
        geom.top_polys.append(mc.rect_polygon("feed_stub", stub_x0, stub_x1, y0, y1))

        gap_x0 = feed_offset + max(feed_in_len, 0.0)
        gap_x1 = gap_x0 + source_gap
        z0 = float(cfg["substrate_thickness_mm"])
        z1 = z0 + gap_height_mm
        geom.meta["port_defs"] = [
            {
                "start": [gap_x0, y0, z0],
                "stop": [gap_x1, y1, z1],
                "exc_dir": "x",
                "R": 50,
                "port_type": port_type,
            }
        ]
        geom.feed_point = (0.5 * (gap_x0 + gap_x1), y_center)
    elif port_mode == "fdtdx":
        feed_in_len = feed_length - feed_stub
        if feed_in_len <= 0:
            raise ValueError("feed_length_mm must exceed feed_stub_length_mm")
        geom.top_polys.append(
            mc.rect_polygon("feed_line", feed_offset, feed_offset + feed_in_len, y0, y1)
        )
        stub_x0 = feed_offset + feed_length - feed_stub
        stub_x1 = feed_offset + feed_length + overlap_mm
        geom.top_polys.append(mc.rect_polygon("feed_stub", stub_x0, stub_x1, y0, y1))
        port_len = max(source_gap, resolution_mm)
        fdtdx_source_offset = None
        fdtdx_port_offset = None
        if port_offsets is not None:
            fdtdx_source_offset, fdtdx_port_offset, fdtdx_source_len, _ = port_offsets
            port_len = max(fdtdx_source_len, resolution_mm)
        if port_type == "msl":
            port_x0 = feed_offset
            port_x1 = feed_offset + feed_in_len
        else:
            port_len = min(port_len, feed_in_len)
            if fdtdx_port_offset is not None:
                port_x0 = feed_offset + fdtdx_port_offset
                port_x1 = port_x0 + port_len
                if port_x1 > feed_offset + feed_in_len:
                    port_x1 = feed_offset + feed_in_len
                    port_x0 = port_x1 - port_len
            else:
                port_x1 = feed_offset + feed_in_len
                port_x0 = port_x1 - port_len
        z_start = 0.0
        z_stop = float(cfg["substrate_thickness_mm"])
        if port_type == "msl":
            z_start, z_stop = z_stop, z_start
        port_y0 = y0
        port_y1 = y1
        if port_type != "msl" and port_width > feed_width:
            port_y0 = y_center - port_width * 0.5
            port_y1 = y_center + port_width * 0.5
        port_def = {
            "start": [port_x0, port_y0, z_start],
            "stop": [port_x1, port_y1, z_stop],
            "exc_dir": "z",
            "R": 50,
            "port_type": port_type,
            "prop_dir": "x",
        }
        if port_type == "msl" and fdtdx_source_offset is not None and fdtdx_port_offset is not None:
            port_def["feed_shift"] = fdtdx_source_offset
            port_def["measplane_shift"] = fdtdx_port_offset
        geom.meta["port_defs"] = [port_def]
        geom.feed_point = (0.5 * (port_x0 + port_x1), y_center)
    elif port_mode == "fdtdx_line":
        feed_in_len = feed_length - feed_stub
        if feed_in_len <= 0:
            raise ValueError("feed_length_mm must exceed feed_stub_length_mm")
        geom.top_polys.append(
            mc.rect_polygon("feed_line", feed_offset, feed_offset + feed_in_len, y0, y1)
        )
        stub_x0 = feed_offset + feed_length - feed_stub
        stub_x1 = feed_offset + feed_length + overlap_mm
        geom.top_polys.append(mc.rect_polygon("feed_stub", stub_x0, stub_x1, y0, y1))
        port_len = max(source_gap, resolution_mm)
        port_len = min(port_len, feed_in_len)
        x_port = feed_offset + feed_in_len - 0.5 * port_len
        if port_offsets is not None:
            _, fdtdx_port_offset, fdtdx_source_len, _ = port_offsets
            port_len = min(max(fdtdx_source_len, resolution_mm), feed_in_len)
            x_port = feed_offset + fdtdx_port_offset + 0.5 * port_len
        z_start = 0.0
        z_stop = float(cfg["substrate_thickness_mm"])
        if port_type == "msl":
            z_start, z_stop = z_stop, z_start
        geom.meta["port_defs"] = [
            {
                "start": [x_port, y_center, z_start],
                "stop": [x_port, y_center, z_stop],
                "exc_dir": "z",
                "R": 50,
                "port_type": port_type,
                "prop_dir": "x",
            }
        ]
        geom.feed_point = (x_port, y_center)
    else:
        geom.top_polys.append(mc.rect_polygon("feed_line", feed_offset, feed_offset + feed_length, y0, y1))
        geom.feed_point = (feed_offset + 0.5 * feed_length, y_center)

    return geom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdtdx-run", required=True, help="FDTDX run directory (contains params.json/geometry.npz)")
    parser.add_argument("--quality", choices=["fast", "medium", "high"], default="medium")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--port-mode", choices=["ground", "gap", "fdtdx", "fdtdx_line"], default="ground")
    parser.add_argument("--port-type", choices=["lumped", "msl"], default="lumped")
    parser.add_argument("--gap-height-mm", type=float, default=0.2)
    parser.add_argument("--no-clip-to-wedge", action="store_true")
    parser.add_argument("--mirror-fdtdx", action="store_true", help="Use FDTDX material thickness/loss settings")
    parser.add_argument("--tan-delta", type=float, default=None, help="Override substrate loss tangent")
    parser.add_argument("--copper-thickness-mm", type=float, default=None, help="Override copper thickness")
    parser.add_argument("--max-cell-mm", type=float, default=None, help="Override mesh max cell size (mm)")
    parser.add_argument("--snap-mm", type=float, default=None, help="Override mesh snap grid (mm)")
    parser.add_argument("--smooth-ratio", type=float, default=None, help="Override mesh smoothing ratio")
    parser.add_argument("--nr-ts", type=int, default=None, help="Override maximum timesteps")
    parser.add_argument("--end-criteria", type=float, default=None, help="Override end criteria threshold")
    parser.add_argument(
        "--match-fdtdx-mesh",
        action="store_true",
        help="Match openEMS mesh max cell + snap to FDTDX resolution",
    )
    parser.add_argument(
        "--overlap-mm",
        type=float,
        default=0.0,
        help="Extend the feed stub into the design region (mm) to ensure connectivity",
    )
    parser.add_argument("--out-dir", default=None, help="Output openEMS run directory")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="Compute NF2FF metrics at f0")
    parser.add_argument(
        "--plane-phi-step",
        type=float,
        default=5.0,
        help="Phi step (deg) for in-plane NF2FF metrics sweep",
    )
    parser.add_argument(
        "--phi-forward",
        type=float,
        default=0.0,
        help="Forward direction (deg) for in-plane NF2FF metrics",
    )
    parser.add_argument("--vtk", action="store_true")
    parser.add_argument("--no-vtk", action="store_true", help="Disable VTK export")
    parser.add_argument("--vtk-out", default=None, help="Optional VTK output path")
    parser.add_argument("--match3", action="store_true", help="Evaluate 3-element (pi) match network")
    parser.add_argument("--match3-l-min-nh", type=float, default=0.5)
    parser.add_argument("--match3-l-max-nh", type=float, default=20.0)
    parser.add_argument("--match3-c-min-pf", type=float, default=0.2)
    parser.add_argument("--match3-c-max-pf", type=float, default=10.0)
    parser.add_argument("--match3-samples", type=int, default=8)
    parser.add_argument("--prune", action="store_true", help="Remove large openEMS artifacts after export")
    args = parser.parse_args()

    mc.require_openems()
    run_dir = os.path.abspath(args.fdtdx_run)
    params, design_indices, resolution_mm = _load_fdtdx_run(run_dir)
    cfg = params["model_cfg"]
    cfg = dict(cfg)
    cfg["resolution_mm"] = resolution_mm

    copper_nominal_mm = float(cfg.get("copper_thickness_mm", 0.035))
    effective_copper_mm = max(copper_nominal_mm, resolution_mm)
    gap_height_mm = float(args.gap_height_mm)
    if args.mirror_fdtdx:
        gap_height_mm = max(gap_height_mm, float(cfg["substrate_thickness_mm"]))
    else:
        gap_height_mm = max(gap_height_mm, copper_nominal_mm)
    if args.port_type == "msl" and args.port_mode != "fdtdx":
        raise ValueError("MSL port is only supported with --port-mode fdtdx")
    overlap_mm = max(args.overlap_mm, 0.0)
    overlap_mm = min(overlap_mm, resolution_mm)
    geom = _build_geometry(
        cfg,
        design_indices,
        args.threshold,
        args.port_mode,
        args.port_type,
        gap_height_mm,
        clip_to_wedge=not args.no_clip_to_wedge,
        overlap_mm=overlap_mm,
    )
    constraints = mc.SliceConstraints(
        slice_angle_deg=float(cfg["slice_angle_deg"]),
        outer_radius_mm=float(cfg["outer_radius_mm"]),
        inner_radius_mm=float(cfg["inner_radius_mm"]),
        keepout_edge_mm=2.0,
        feed_location_mm=float(cfg["feed_offset_mm"] + cfg["feed_length_mm"] * 0.5),
        min_trace_mm=float(cfg.get("min_trace_mm", 0.2)),
        min_gap_mm=float(cfg.get("min_gap_mm", 0.2)),
    )
    eps_r = float(cfg.get("eps_r", 4.4))
    tan_delta = 0.02
    copper_thickness = 0.035
    if args.mirror_fdtdx:
        tan_delta = 0.0
        copper_thickness = effective_copper_mm
    if args.tan_delta is not None:
        tan_delta = args.tan_delta
    if args.copper_thickness_mm is not None:
        copper_thickness = args.copper_thickness_mm
        if args.mirror_fdtdx:
            copper_thickness = max(copper_thickness, resolution_mm)

    substrate = mc.SubstrateConfig(
        eps_r=eps_r,
        tan_delta=tan_delta,
        thickness_mm=float(cfg["substrate_thickness_mm"]),
        copper_thickness_mm=copper_thickness,
    )

    f0 = float(cfg["f0_hz"])
    fmin = float(cfg["f_low_hz"])
    fmax = float(cfg["f_high_hz"])
    quality = mc.quality_from_name(args.quality, fmax, substrate.eps_r, constraints.min_trace_mm)
    smooth_ratio = quality.smooth_ratio
    if args.smooth_ratio is not None:
        smooth_ratio = float(args.smooth_ratio)
    if args.match_fdtdx_mesh:
        quality = mc.QualitySettings(
            name=quality.name,
            max_cell_mm=resolution_mm,
            end_criteria=quality.end_criteria,
            air_margin_mm=quality.air_margin_mm,
            smooth_ratio=smooth_ratio,
            nr_ts=quality.nr_ts,
            snap_mm=resolution_mm,
        )
    if args.max_cell_mm is not None or args.snap_mm is not None:
        quality = mc.QualitySettings(
            name=quality.name,
            max_cell_mm=float(args.max_cell_mm) if args.max_cell_mm is not None else quality.max_cell_mm,
            end_criteria=quality.end_criteria,
            air_margin_mm=quality.air_margin_mm,
            smooth_ratio=smooth_ratio,
            nr_ts=quality.nr_ts,
            snap_mm=float(args.snap_mm) if args.snap_mm is not None else quality.snap_mm,
        )
    if args.end_criteria is not None or args.nr_ts is not None:
        quality = mc.QualitySettings(
            name=quality.name,
            max_cell_mm=quality.max_cell_mm,
            end_criteria=float(args.end_criteria) if args.end_criteria is not None else quality.end_criteria,
            air_margin_mm=quality.air_margin_mm,
            smooth_ratio=smooth_ratio,
            nr_ts=int(args.nr_ts) if args.nr_ts is not None else quality.nr_ts,
            snap_mm=quality.snap_mm,
        )
    fc = max((fmax - fmin) * 2.0, f0 * 0.25)

    payload = {
        "fdtdx_run": run_dir,
        "model": params.get("model"),
        "f0_hz": f0,
        "f_low_hz": fmin,
        "f_high_hz": fmax,
        "band_hz": fmax - fmin,
        "quality": args.quality,
        "threshold": args.threshold,
        "port_mode": args.port_mode,
        "port_type": args.port_type,
        "gap_height_mm": gap_height_mm,
        "overlap_mm": overlap_mm,
        "mirror_fdtdx": args.mirror_fdtdx,
        "match_fdtdx_mesh": args.match_fdtdx_mesh,
        "match3": args.match3,
        "match3_l_min_nh": args.match3_l_min_nh,
        "match3_l_max_nh": args.match3_l_max_nh,
        "match3_c_min_pf": args.match3_c_min_pf,
        "match3_c_max_pf": args.match3_c_max_pf,
        "match3_samples": args.match3_samples,
        "plane_phi_step_deg": args.plane_phi_step,
        "plane_phi_forward_deg": args.phi_forward,
        "tan_delta": tan_delta,
        "copper_thickness_mm": copper_thickness,
        "clip_to_wedge": not args.no_clip_to_wedge,
        "max_cell_mm": quality.max_cell_mm,
        "snap_mm": quality.snap_mm,
        "smooth_ratio": smooth_ratio,
        "nr_ts": quality.nr_ts,
        "end_criteria": quality.end_criteria,
    }
    if args.port_mode in ("fdtdx", "fdtdx_line"):
        try:
            src_off, port_off, source_len, det_len = _fdtdx_port_offsets_mm(cfg, resolution_mm)
            payload["port_source_offset_mm"] = src_off
            payload["port_offset_mm"] = port_off
            payload["port_source_len_mm"] = source_len
            payload["port_det_len_mm"] = det_len
        except Exception:
            pass
    run_hash = mc.hash_params(payload)
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "runs_fdtdx", run_hash)
    os.makedirs(out_dir, exist_ok=True)

    _export_geometry(out_dir, geom)

    mc.save_json(os.path.join(out_dir, "fdtdx_meta.json"), payload)
    mc.save_json(os.path.join(out_dir, "params.json"), {"fdtdx": params, "substrate": substrate.__dict__})

    FDTD, ports, nf2ff, meta = mc.build_simulation(
        geom,
        substrate,
        constraints,
        quality,
        f0_hz=f0,
        fc_hz=fc,
        fmax_hz=fmax,
        excite_port=0,
        port_count=1,
    )

    log_path = os.path.join(out_dir, "openEMS_log.txt")
    has_timeseries = any(
        os.path.exists(os.path.join(out_dir, name))
        for name in ("port_ut_1A", "port_ut_1B", "port_it_1A", "port_it_1B")
    )
    if args.force or not (os.path.exists(log_path) or has_timeseries):
        FDTD.Run(out_dir, cleanup=args.force, verbose=0)
        mc.save_json(os.path.join(out_dir, "fdtdx_meta.json"), payload)
        mc.save_json(os.path.join(out_dir, "params.json"), {"fdtdx": params, "substrate": substrate.__dict__})

    freq_plot = np.linspace(0.1e9, 6.0e9, 901)
    post.calc_sparams(ports, out_dir, freq_plot, ref_impedance=50, excite_port=0)
    s11, zin = post.port_s11_zin(ports[0])
    s11_db = post.s11_db(s11)
    post.save_s11_csv(os.path.join(out_dir, "s11.csv"), freq_plot, s11_db, zin)
    post.save_s11_plot(os.path.join(out_dir, "s11.png"), freq_plot, s11_db)

    match_meta = None
    s11_matched_db = None
    zin_matched = None
    s11_matched = None
    try:
        zin_f0 = zin[int(np.argmin(np.abs(freq_plot - f0)))]
        match_meta = matching.calc_l_match(zin_f0, f0)
    except Exception:
        match_meta = None

    if match_meta:
        zin_matched, s11_matched, s11_matched_db = matching.apply_l_match(
            zin, freq_plot, match_meta, z0=50.0
        )
        post.save_s11_csv(os.path.join(out_dir, "s11_matched.csv"), freq_plot, s11_matched_db, zin_matched)
        post.save_s11_plot(os.path.join(out_dir, "s11_matched.png"), freq_plot, s11_matched_db)
        post.save_smith_csv(os.path.join(out_dir, "s11_matched_smith.csv"), freq_plot, s11_matched)

    match3_meta = None
    if args.match3:
        match3_meta = matching.calc_pi_match(
            zin,
            freq_plot,
            f0,
            fmin,
            fmax,
            z0=50.0,
            l_range_nh=(args.match3_l_min_nh, args.match3_l_max_nh),
            c_range_pf=(args.match3_c_min_pf, args.match3_c_max_pf),
            samples=max(2, int(args.match3_samples)),
        )
        if match3_meta:
            post.save_s11_csv(os.path.join(out_dir, "s11_matched3.csv"), freq_plot, match3_meta["s11_db"], match3_meta["zin"])
            post.save_s11_plot(os.path.join(out_dir, "s11_matched3.png"), freq_plot, match3_meta["s11_db"])
            post.save_smith_csv(os.path.join(out_dir, "s11_matched3_smith.csv"), freq_plot, match3_meta["s11"])

    metrics_out = None
    nf2ff_full = None
    nf2ff_theta = None
    nf2ff_phi = None
    if args.metrics:
        band_mask = (freq_plot >= fmin) & (freq_plot <= fmax)
        if np.any(band_mask):
            rl_band = -s11_db[band_mask]
            freq_band = freq_plot[band_mask]
            peak_idx = int(np.argmax(rl_band))
            metrics_out = {
                "rl_min_in_band_db": float(np.min(rl_band)),
                "rl_peak_in_band_db": float(rl_band[peak_idx]),
                "f_peak_in_band_hz": float(freq_band[peak_idx]),
            }
        else:
            metrics_out = {
                "rl_min_in_band_db": float("nan"),
                "rl_peak_in_band_db": float("nan"),
                "f_peak_in_band_hz": float("nan"),
            }
        f0_idx = int(np.argmin(np.abs(freq_plot - f0)))
        zin_f0 = zin[f0_idx]
        metrics_out["zin_f0_real_ohm"] = float(np.real(zin_f0)) if np.isfinite(zin_f0) else float("nan")
        metrics_out["zin_f0_imag_ohm"] = float(np.imag(zin_f0)) if np.isfinite(zin_f0) else float("nan")
        metrics_out["f0_hz"] = f0
        metrics_out["f_low_hz"] = fmin
        metrics_out["f_high_hz"] = fmax
        metrics_out["band_hz"] = fmax - fmin
        metrics_out["model"] = params.get("model")

        if s11_matched_db is not None and match_meta:
            rl_matched = -s11_matched_db
            band_mask = (freq_plot >= fmin) & (freq_plot <= fmax)
            if np.any(band_mask):
                rl_band = rl_matched[band_mask]
                freq_band = freq_plot[band_mask]
                peak_idx = int(np.argmax(rl_band))
                metrics_out["rl_min_in_band_matched_db"] = float(np.min(rl_band))
                metrics_out["rl_peak_in_band_matched_db"] = float(rl_band[peak_idx])
                metrics_out["f_peak_in_band_matched_hz"] = float(freq_band[peak_idx])
            else:
                metrics_out["rl_min_in_band_matched_db"] = float("nan")
                metrics_out["rl_peak_in_band_matched_db"] = float("nan")
                metrics_out["f_peak_in_band_matched_hz"] = float("nan")

            bw_hz, bw_frac = matching.matched_bandwidth(
                freq_plot,
                rl_matched,
                fmin,
                fmax,
                rl_target=10.0,
            )
            series_val = float(match_meta["series_value"])
            shunt_val = float(match_meta["shunt_value"])
            metrics_out.update(
                {
                    "match_topology": match_meta["topology"],
                    "match_series_type": match_meta["series_type"],
                    "match_series_value": series_val,
                    "match_shunt_type": match_meta["shunt_type"],
                    "match_shunt_value": shunt_val,
                    "match_series_value_nh": series_val * 1e9 if match_meta["series_type"] == "L" else float("nan"),
                    "match_series_value_pf": series_val * 1e12 if match_meta["series_type"] == "C" else float("nan"),
                    "match_shunt_value_nh": shunt_val * 1e9 if match_meta["shunt_type"] == "L" else float("nan"),
                    "match_shunt_value_pf": shunt_val * 1e12 if match_meta["shunt_type"] == "C" else float("nan"),
                    "match_penalty": float(match_meta["penalty"]),
                    "match_bandwidth_hz": bw_hz,
                    "match_bandwidth_frac": bw_frac,
                }
            )

        if match3_meta:
            rl_match3 = -match3_meta["s11_db"]
            band_mask = (freq_plot >= fmin) & (freq_plot <= fmax)
            if np.any(band_mask):
                rl_band = rl_match3[band_mask]
                freq_band = freq_plot[band_mask]
                peak_idx = int(np.argmax(rl_band))
                metrics_out["match3_rl_min_in_band_db"] = float(np.min(rl_band))
                metrics_out["match3_rl_peak_in_band_db"] = float(rl_band[peak_idx])
                metrics_out["match3_f_peak_in_band_hz"] = float(freq_band[peak_idx])
            else:
                metrics_out["match3_rl_min_in_band_db"] = float("nan")
                metrics_out["match3_rl_peak_in_band_db"] = float("nan")
                metrics_out["match3_f_peak_in_band_hz"] = float("nan")

            metrics_out.update(
                {
                    "match3_topology": match3_meta["topology"],
                    "match3_shunt_in_type": match3_meta["shunt_in_type"],
                    "match3_shunt_in_value": float(match3_meta["shunt_in_value"]),
                    "match3_series_type": match3_meta["series_type"],
                    "match3_series_value": float(match3_meta["series_value"]),
                    "match3_shunt_out_type": match3_meta["shunt_out_type"],
                    "match3_shunt_out_value": float(match3_meta["shunt_out_value"]),
                    "match3_bandwidth_hz": float(match3_meta["match_bandwidth_hz"]),
                    "match3_bandwidth_frac": float(match3_meta["match_bandwidth_frac"]),
                }
            )
        try:
            from openEMS import nf2ff as nf2ff_mod
            from CSXCAD import ContinuousStructure
        except Exception:
            print(
                "openEMS Python modules not available.\n"
                "Activate your openEMS environment first (for example):\n"
                "  source ../scripts/env.sh"
            )
        else:
            CSX = ContinuousStructure()
            nf2ff_box = nf2ff_mod.nf2ff(CSX, "nf2ff", meta["nf2ff_start_mm"], meta["nf2ff_stop_mm"])
            phi_step = max(1.0, float(args.plane_phi_step))
            theta_grid = np.arange(0.0, 180.0 + phi_step, phi_step)
            phi_grid = np.arange(0.0, 360.0 + phi_step, phi_step)
            res = nf2ff_box.CalcNF2FF(
                out_dir,
                f0,
                theta_grid.tolist(),
                phi_grid.tolist(),
                radius=1,
                center=[0, 0, 0],
                outfile="nf2ff_metrics.h5",
                read_cached=False,
            )
            nf2ff_full = res
            nf2ff_theta = theta_grid
            nf2ff_phi = phi_grid
            p_inc = float(post.port_power_inc_at(ports[0], freq_plot, f0))
            p_ref = float(post.port_power_ref_at(ports[0], freq_plot, f0))
            p_acc = float(post.port_power_acc_at(ports[0], freq_plot, f0))
            prad = float(res.Prad[0])
            scale = 1.0
            if math.isfinite(p_inc) and p_inc > 0:
                scale = 1.0 / p_inc
            p_inc_norm = p_inc * scale
            p_ref_norm = p_ref * scale
            p_acc_norm = p_acc * scale
            prad_norm = prad * scale
            p_acc_min = 1e-3
            valid_pacc = math.isfinite(p_acc_norm) and p_acc_norm > p_acc_min

            gain_plane_db = None
            gain_plane_realized_db = None
            phi_forward = float(args.phi_forward) % 360.0
            phi_peak = float("nan")
            phi_delta = float("nan")
            gain_peak = float("nan")
            gain_peak_realized = float("nan")
            fb_peak = float("nan")
            gain_fwd = float("nan")
            gain_back = float("nan")
            fb_forward = float("nan")
            gain_fwd_realized = float("nan")
            gain_back_realized = float("nan")
            prad_ok = math.isfinite(prad_norm) and prad_norm > 0
            if prad_ok:
                u_grid = np.asarray(res.P_rad[0], dtype=np.float64) * scale
                directivity = 4.0 * math.pi * u_grid / prad_norm
                gain_grid_db = 10.0 * np.log10(np.maximum(directivity, 1e-12))
                if valid_pacc:
                    realized = 4.0 * math.pi * u_grid / p_acc_norm
                    gain_grid_realized_db = 10.0 * np.log10(np.maximum(realized, 1e-12))
                else:
                    gain_grid_realized_db = None

                theta_idx = int(np.argmin(np.abs(theta_grid - 90.0)))
                gain_plane_db = gain_grid_db[theta_idx, :]
                peak_idx = int(np.argmax(gain_plane_db))
                phi_peak = float(phi_grid[peak_idx] % 360.0)
                gain_peak = float(gain_plane_db[peak_idx])
                if gain_grid_realized_db is not None:
                    gain_plane_realized_db = gain_grid_realized_db[theta_idx, :]
                    gain_peak_realized = float(gain_plane_realized_db[peak_idx])

                phi_back = (phi_peak + 180.0) % 360.0
                back_idx = int(np.argmin(np.abs(phi_grid - phi_back)))
                fb_peak = gain_peak - float(gain_plane_db[back_idx])

                fwd_idx = int(np.argmin(np.abs(phi_grid - phi_forward)))
                gain_fwd = float(gain_plane_db[fwd_idx])
                phi_back_fwd = (phi_forward + 180.0) % 360.0
                back_fwd_idx = int(np.argmin(np.abs(phi_grid - phi_back_fwd)))
                gain_back = float(gain_plane_db[back_fwd_idx])
                fb_forward = gain_fwd - gain_back
                if gain_plane_realized_db is not None:
                    gain_fwd_realized = float(gain_plane_realized_db[fwd_idx])
                    gain_back_realized = float(gain_plane_realized_db[back_fwd_idx])

                delta = abs(phi_peak - phi_forward)
                phi_delta = min(delta, 360.0 - delta)

            rad_eff = float("nan")
            eff_tol = 1.05
            eff_ok = valid_pacc and math.isfinite(prad_norm) and prad_norm >= 0 and prad_norm <= p_acc_norm * eff_tol
            if eff_ok:
                rad_eff = prad_norm / p_acc_norm if p_acc_norm > 0 else float("nan")

            lam = 299792458.0 / f0
            gain_lin = 10.0 ** (gain_fwd_realized / 10.0) if math.isfinite(gain_fwd_realized) else float("nan")
            eff_ap = (lam * lam / (4.0 * math.pi)) * gain_lin if math.isfinite(gain_lin) else float("nan")
            metrics_out.update(
                {
                    "gain_fwd_db": round(gain_fwd, 3) if math.isfinite(gain_fwd) else float("nan"),
                    "fb_db": round(fb_forward, 3) if math.isfinite(fb_forward) else float("nan"),
                    "gain_fwd_realized_db": round(gain_fwd_realized, 3) if math.isfinite(gain_fwd_realized) else float("nan"),
                    "gain_back_realized_db": round(gain_back_realized, 3)
                    if math.isfinite(gain_back_realized)
                    else float("nan"),
                    "plane_phi_step_deg": float(phi_step),
                    "plane_phi_forward_deg": float(phi_forward),
                    "plane_phi_peak_deg": round(phi_peak, 3) if math.isfinite(phi_peak) else float("nan"),
                    "plane_phi_delta_deg": round(phi_delta, 3) if math.isfinite(phi_delta) else float("nan"),
                    "plane_gain_peak_db": round(gain_peak, 3) if math.isfinite(gain_peak) else float("nan"),
                    "plane_gain_peak_realized_db": round(gain_peak_realized, 3)
                    if math.isfinite(gain_peak_realized)
                    else float("nan"),
                    "plane_fb_peak_db": round(fb_peak, 3) if math.isfinite(fb_peak) else float("nan"),
                    "plane_gain_forward_db": round(gain_fwd, 3) if math.isfinite(gain_fwd) else float("nan"),
                    "plane_gain_forward_realized_db": round(gain_fwd_realized, 3)
                    if math.isfinite(gain_fwd_realized)
                    else float("nan"),
                    "plane_fb_forward_db": round(fb_forward, 3) if math.isfinite(fb_forward) else float("nan"),
                    "prad_w": float(prad_norm) if math.isfinite(prad_norm) else float("nan"),
                    "p_inc_w": float(p_inc_norm) if math.isfinite(p_inc_norm) else float("nan"),
                    "p_ref_w": float(p_ref_norm) if math.isfinite(p_ref_norm) else float("nan"),
                    "p_acc_w": float(p_acc_norm) if math.isfinite(p_acc_norm) else float("nan"),
                    "p_acc_sign": float(math.copysign(1.0, p_acc_norm)) if math.isfinite(p_acc_norm) and p_acc_norm != 0 else float("nan"),
                    "p_inc_w_raw": float(p_inc) if math.isfinite(p_inc) else float("nan"),
                    "p_ref_w_raw": float(p_ref) if math.isfinite(p_ref) else float("nan"),
                    "p_acc_w_raw": float(p_acc) if math.isfinite(p_acc) else float("nan"),
                    "rad_eff": round(rad_eff, 6) if math.isfinite(rad_eff) else float("nan"),
                    "rad_eff_pct": round(rad_eff * 100.0, 3) if math.isfinite(rad_eff) else float("nan"),
                    "effective_aperture_m2": round(eff_ap, 8) if math.isfinite(eff_ap) else float("nan"),
                    "valid_power": bool(valid_pacc and prad_ok and eff_ok),
                }
            )

    mc.save_json(os.path.join(out_dir, "meta.json"), {"nf2ff": meta, "fdtdx": params})
    if metrics_out is not None:
        mc.save_json(os.path.join(out_dir, "metrics.json"), metrics_out)

    do_vtk = args.vtk or not args.no_vtk
    exit_code = 0
    if do_vtk:
        try:
            from openEMS import nf2ff as nf2ff_mod
            from CSXCAD import ContinuousStructure
        except Exception:
            print(
                "openEMS Python modules not available.\n"
                "Activate your openEMS environment first (for example):\n"
                "  source ../scripts/env.sh"
            )
            exit_code = 1
        vtk_out = args.vtk_out or os.path.join(out_dir, "pattern_3d.vtk")
        nf2ff_meta = meta
        start = nf2ff_meta.get("nf2ff_start_mm")
        stop = nf2ff_meta.get("nf2ff_stop_mm")
        if not start or not stop:
            print("nf2ff metadata missing; cannot generate VTK.")
            exit_code = 1
        if exit_code == 0:
            theta = nf2ff_theta
            phi = nf2ff_phi
            res = nf2ff_full
            if res is None or theta is None or phi is None:
                theta = np.arange(0.0, 180.0 + 5.0, 5.0)
                phi = np.arange(0.0, 360.0 + 5.0, 5.0)
                CSX = ContinuousStructure()
                nf2ff_box = nf2ff_mod.nf2ff(CSX, "nf2ff", start, stop)
                res = nf2ff_box.CalcNF2FF(
                    out_dir,
                    f0,
                    theta,
                    phi,
                    radius=1,
                    center=[0, 0, 0],
                    outfile=os.path.basename(vtk_out) + ".h5",
                    read_cached=False,
                )
            prad = float(res.Prad[0])
            if prad <= 0:
                print("Invalid radiated power in NF2FF result.")
                exit_code = 1
            else:
                u = res.P_rad[0]
                directivity = 4.0 * math.pi * u / prad
                gain_db = 10.0 * np.log10(np.maximum(directivity, 1e-12))
                os.makedirs(os.path.dirname(vtk_out), exist_ok=True)
                visualize_pattern.write_vtk(vtk_out, theta, phi, gain_db)
                print(vtk_out)

    if args.prune:
        _prune_run(out_dir)

    print(out_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

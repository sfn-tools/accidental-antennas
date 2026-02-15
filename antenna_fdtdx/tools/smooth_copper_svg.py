#!/usr/bin/env python3
"""
Smooth tile-like copper SVG geometry into clean outline paths.

Default behavior:
- Run with only an input file: the output is written next to it as
  "<input_stem>_smooth.svg".
- Defaults are tuned for coarse FDTDX tile exports and match tested settings:
  gaussian smoothing with radius 1.2, scale 24, simplify 0.10, and Chaikin
  corner-rounding 3 iterations.

Optional aligned ground workflow:
- Pass --ground-svg to smooth top and ground together and also write an aligned
  composite SVG with both layers in one shared frame.
- The alignment uses run metadata from sibling params.json (fdtdx.model_cfg)
  and is tailored to FDTDX/openEMS bridge exports.

The script is aimed at FDTDX/OpenEMS copper exports where geometry is stored as
many small polygons. It rasterizes polygons, applies smoothing, then vectorizes
back to SVG paths that import cleanly into KiCad.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_length(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing length attribute.")
    match = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", value)
    if not match:
        raise ValueError(f"Could not parse length: {value!r}")
    return float(match.group(1))


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    vb = root.get("viewBox")
    if vb:
        nums = [float(x) for x in re.split(r"[\s,]+", vb.strip()) if x]
        if len(nums) == 4:
            return nums[0], nums[1], nums[2], nums[3]
        raise ValueError(f"Invalid viewBox: {vb!r}")
    width = _parse_length(root.get("width"))
    height = _parse_length(root.get("height"))
    return 0.0, 0.0, width, height


def _parse_points(points_text: str) -> np.ndarray:
    nums = [float(x) for x in re.split(r"[\s,]+", points_text.strip()) if x]
    if len(nums) < 6 or (len(nums) % 2) != 0:
        raise ValueError(f"Invalid polygon points: {points_text[:80]!r}")
    pts = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.float64)
    return pts


def _extract_polygons(root: ET.Element) -> list[np.ndarray]:
    polys: list[np.ndarray] = []
    for elem in root.iter():
        tag = _local_name(elem.tag)
        if tag == "polygon":
            points_text = elem.get("points", "")
            if not points_text.strip():
                continue
            polys.append(_parse_points(points_text))
        elif tag == "rect":
            x = _parse_length(elem.get("x", "0"))
            y = _parse_length(elem.get("y", "0"))
            w = _parse_length(elem.get("width"))
            h = _parse_length(elem.get("height"))
            pts = np.array(
                [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ],
                dtype=np.float64,
            )
            polys.append(pts)
    return polys


def _estimate_grid_step(polygons: list[np.ndarray]) -> float | None:
    if not polygons:
        return None
    xs = np.unique(np.concatenate([p[:, 0] for p in polygons]))
    ys = np.unique(np.concatenate([p[:, 1] for p in polygons]))
    diffs = []
    if xs.size > 1:
        dx = np.diff(np.sort(xs))
        diffs.extend([float(v) for v in dx if v > 1e-9])
    if ys.size > 1:
        dy = np.diff(np.sort(ys))
        diffs.extend([float(v) for v in dy if v > 1e-9])
    if not diffs:
        return None
    diffs = np.array(diffs, dtype=np.float64)
    return float(np.percentile(diffs, 10))


def _rasterize(
    polygons: list[np.ndarray],
    vb: tuple[float, float, float, float],
    scale: float,
) -> np.ndarray:
    vb_x, vb_y, vb_w, vb_h = vb
    width_px = max(1, int(math.ceil(vb_w * scale)))
    height_px = max(1, int(math.ceil(vb_h * scale)))
    mask = np.zeros((height_px, width_px), dtype=np.uint8)

    for poly in polygons:
        pts = np.empty((poly.shape[0], 1, 2), dtype=np.int32)
        x_px = np.round((poly[:, 0] - vb_x) * scale).astype(np.int64)
        y_px = np.round((poly[:, 1] - vb_y) * scale).astype(np.int64)
        x_px = np.clip(x_px, 0, width_px - 1)
        y_px = np.clip(y_px, 0, height_px - 1)
        pts[:, 0, 0] = x_px.astype(np.int32)
        pts[:, 0, 1] = y_px.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def _smooth_mask(mask: np.ndarray, radius_px: int, mode: str) -> np.ndarray:
    if radius_px <= 0:
        return mask

    if mode == "gaussian":
        sigma = max(0.8, radius_px / 2.0)
        ksize = int(2 * round(3.0 * sigma) + 1)
        ksize = max(3, ksize)
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return out

    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if mode == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if mode == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if mode == "close-open":
        tmp = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
    if mode == "open-close":
        tmp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)

    raise ValueError(f"Unsupported smoothing mode: {mode}")


def _chaikin_closed(points: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0 or points.shape[0] < 3:
        return points
    out = points.astype(np.float64, copy=True)
    for _ in range(iters):
        n = out.shape[0]
        if n < 3:
            break
        nxt = np.roll(out, -1, axis=0)
        q = 0.75 * out + 0.25 * nxt
        r = 0.25 * out + 0.75 * nxt
        interleaved = np.empty((2 * n, 2), dtype=np.float64)
        interleaved[0::2] = q
        interleaved[1::2] = r
        out = interleaved
    return out


def _contour_to_subpath(
    contour: np.ndarray,
    vb: tuple[float, float, float, float],
    scale: float,
    simplify_px: float,
    chaikin_iters: int,
) -> str:
    points = contour[:, 0, :].astype(np.float64)

    if simplify_px > 0:
        contour = cv2.approxPolyDP(contour, epsilon=simplify_px, closed=True)
        points = contour[:, 0, :].astype(np.float64)

    points = _chaikin_closed(points, chaikin_iters)

    if points.shape[0] < 3:
        return ""

    vb_x, vb_y, _, _ = vb
    parts = []
    for i, (x_px, y_px) in enumerate(points):
        x = vb_x + float(x_px) / scale
        y = vb_y + float(y_px) / scale
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd} {x:.4f} {y:.4f}")
    parts.append("Z")
    return " ".join(parts)


def _vectorize(
    mask: np.ndarray,
    vb: tuple[float, float, float, float],
    scale: float,
    simplify: float,
    min_island_area: float,
    chaikin_iters: int,
) -> str:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return ""

    hierarchy = hierarchy[0]
    min_area_px2 = max(0.0, min_island_area * (scale * scale))
    simplify_px = max(0.0, simplify * scale)

    children_by_parent: dict[int, list[int]] = {}
    for idx, h in enumerate(hierarchy):
        parent = int(h[3])
        if parent >= 0:
            children_by_parent.setdefault(parent, []).append(idx)

    paths: list[str] = []
    for idx, h in enumerate(hierarchy):
        parent = int(h[3])
        if parent != -1:
            continue
        outer = contours[idx]
        if abs(cv2.contourArea(outer)) < min_area_px2:
            continue
        subpaths = []
        d_outer = _contour_to_subpath(
            outer,
            vb,
            scale,
            simplify_px,
            chaikin_iters=chaikin_iters,
        )
        if d_outer:
            subpaths.append(d_outer)
        for child_idx in children_by_parent.get(idx, []):
            hole = contours[child_idx]
            if abs(cv2.contourArea(hole)) < min_area_px2:
                continue
            d_hole = _contour_to_subpath(
                hole,
                vb,
                scale,
                simplify_px,
                chaikin_iters=chaikin_iters,
            )
            if d_hole:
                subpaths.append(d_hole)
        if subpaths:
            paths.append(" ".join(subpaths))

    return " ".join(paths)


def _write_svg(
    out_path: Path,
    width_attr: str | None,
    height_attr: str | None,
    vb: tuple[float, float, float, float],
    path_d: str,
) -> None:
    vb_x, vb_y, vb_w, vb_h = vb
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "version": "1.1",
            "viewBox": f"{vb_x} {vb_y} {vb_w} {vb_h}",
            "width": width_attr if width_attr else f"{vb_w}",
            "height": height_attr if height_attr else f"{vb_h}",
        },
    )
    ET.SubElement(
        root,
        "path",
        {
            "d": path_d,
            "fill": "#000000",
            "stroke": "none",
            "fill-rule": "evenodd",
        },
    )
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _write_aligned_svg(
    out_path: Path,
    top_d: str,
    top_vb: tuple[float, float, float, float],
    ground_d: str,
    ground_vb: tuple[float, float, float, float],
    ground_translate: tuple[float, float],
) -> None:
    top_x, top_y, top_w, top_h = top_vb
    g_x, g_y, g_w, g_h = ground_vb
    dx, dy = ground_translate

    x0 = min(top_x, g_x + dx)
    y0 = min(top_y, g_y + dy)
    x1 = max(top_x + top_w, g_x + g_w + dx)
    y1 = max(top_y + top_h, g_y + g_h + dy)
    vb_w = max(1e-6, x1 - x0)
    vb_h = max(1e-6, y1 - y0)

    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "version": "1.1",
            "viewBox": f"{x0:.4f} {y0:.4f} {vb_w:.4f} {vb_h:.4f}",
            "width": f"{vb_w:.4f}",
            "height": f"{vb_h:.4f}",
        },
    )

    ET.SubElement(
        root,
        "path",
        {
            "id": "ground",
            "d": ground_d,
            "fill": "#1F77B4",
            "fill-opacity": "0.45",
            "stroke": "none",
            "fill-rule": "evenodd",
            "transform": f"translate({dx:.4f} {dy:.4f})",
        },
    )
    ET.SubElement(
        root,
        "path",
        {
            "id": "top",
            "d": top_d,
            "fill": "#D62728",
            "fill-opacity": "0.70",
            "stroke": "none",
            "fill-rule": "evenodd",
        },
    )
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _wedge_points(
    inner_r: float,
    outer_r: float,
    angle_deg: float,
    n: int = 48,
) -> list[tuple[float, float]]:
    half = math.radians(angle_deg / 2.0)
    angles_outer = [(-half) + i * (2.0 * half) / (n - 1) for i in range(n)]
    angles_inner = list(reversed(angles_outer))
    outer = [(outer_r * math.cos(a), outer_r * math.sin(a)) for a in angles_outer]
    inner = [(inner_r * math.cos(a), inner_r * math.sin(a)) for a in angles_inner]
    return outer + inner


def _load_model_cfg_near(paths: list[Path]) -> dict | None:
    checked: set[Path] = set()
    for p in paths:
        for cand in [p.parent / "params.json", p.parent.parent / "params.json"]:
            if cand in checked:
                continue
            checked.add(cand)
            if not cand.exists():
                continue
            try:
                payload = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                fd = payload.get("fdtdx")
                if isinstance(fd, dict):
                    cfg = fd.get("model_cfg")
                    if isinstance(cfg, dict):
                        return cfg
                cfg = payload.get("model_cfg")
                if isinstance(cfg, dict):
                    return cfg
    return None


def _poly_bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(poly[:, 0])),
        float(np.max(poly[:, 0])),
        float(np.min(poly[:, 1])),
        float(np.max(poly[:, 1])),
    )


def _infer_feed_center_local_y(top_polys_local: list[np.ndarray]) -> float:
    all_x = np.concatenate([p[:, 0] for p in top_polys_local])
    min_x = float(np.min(all_x))
    tol = 1e-5
    best = None
    for poly in top_polys_local:
        x0, x1, y0, y1 = _poly_bbox(poly)
        if x0 > (min_x + tol):
            continue
        width = x1 - x0
        height = y1 - y0
        y_mid = 0.5 * (y0 + y1)
        cand = (width, -height, y_mid)
        if best is None or cand > best[0]:
            best = (cand, y_mid)
    if best is not None:
        return float(best[1])
    all_y = np.concatenate([p[:, 1] for p in top_polys_local])
    return float(0.5 * (np.min(all_y) + np.max(all_y)))


def _compute_ground_alignment_translation(
    cfg: dict,
    top_polys_local: list[np.ndarray],
    source_scale: float,
    source_margin: float,
) -> tuple[float, float, dict]:
    inner_r = float(cfg["inner_radius_mm"])
    outer_r = float(cfg["outer_radius_mm"])
    angle = float(cfg["slice_angle_deg"])

    full_pts = _wedge_points(inner_r, outer_r, angle)
    full_min_x = min(x for x, _ in full_pts)
    full_min_y = min(y for _, y in full_pts)
    full_shifted = [(x - full_min_x, y - full_min_y) for x, y in full_pts]
    wedge_h = max(y for _, y in full_shifted)

    ground_trim = float(cfg.get("ground_trim_mm", 0.0))
    min_gap = float(cfg.get("min_gap_mm", 0.2))
    if ground_trim > 0:
        min_outer = inner_r + max(min_gap, 0.5)
        ground_outer_r = max(min_outer, outer_r - ground_trim)
    else:
        ground_outer_r = outer_r

    ground_pts = _wedge_points(inner_r, ground_outer_r, angle)
    ground_shifted = [(x - full_min_x, y - full_min_y) for x, y in ground_pts]
    min_x_ground = min(x for x, _ in ground_shifted)
    max_y_ground = max(y for _, y in ground_shifted)

    min_x_top = float(cfg["feed_offset_mm"])
    top_feed_center_local_y = _infer_feed_center_local_y(top_polys_local)
    y_center_world = 0.5 * wedge_h
    max_y_top = y_center_world + (top_feed_center_local_y - source_margin) / source_scale

    delta_x_top_minus_ground = source_scale * (min_x_top - min_x_ground)
    delta_y_top_minus_ground = source_scale * (max_y_ground - max_y_top)
    # Move ground local coords into top-local frame.
    tx = -delta_x_top_minus_ground
    ty = -delta_y_top_minus_ground
    meta = {
        "feed_offset_mm": min_x_top,
        "ground_min_x_mm": min_x_ground,
        "ground_max_y_mm": max_y_ground,
        "top_max_y_mm": max_y_top,
        "top_feed_center_local_y": top_feed_center_local_y,
        "wedge_height_mm": wedge_h,
    }
    return tx, ty, meta


def _process_layer(
    in_path: Path,
    scale: float,
    radius: float,
    mode: str,
    simplify: float,
    min_island_area: float,
    chaikin_iters: int,
    debug_mask_path: Path | None = None,
) -> dict:
    tree = ET.parse(in_path)
    root = tree.getroot()
    width_attr = root.get("width")
    height_attr = root.get("height")
    vb = _parse_viewbox(root)

    polygons = _extract_polygons(root)
    if not polygons:
        raise RuntimeError(f"No <polygon> or <rect> geometry found in {in_path}.")
    grid_step = _estimate_grid_step(polygons)

    mask = _rasterize(polygons, vb, scale)
    radius_px = int(round(radius * scale))
    smoothed = _smooth_mask(mask, radius_px=radius_px, mode=mode)
    if debug_mask_path is not None:
        cv2.imwrite(str(debug_mask_path), smoothed)

    path_d = _vectorize(
        smoothed,
        vb=vb,
        scale=scale,
        simplify=simplify,
        min_island_area=min_island_area,
        chaikin_iters=chaikin_iters,
    )
    if not path_d:
        raise RuntimeError(
            f"Vectorization produced no contours for {in_path}. Try reducing --radius."
        )
    return {
        "in_path": in_path,
        "width_attr": width_attr,
        "height_attr": height_attr,
        "vb": vb,
        "polygons_local": polygons,
        "grid_step": grid_step,
        "path_d": path_d,
        "mask_shape": smoothed.shape,
        "radius_px": radius_px,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Union and smooth tiled copper SVG into a clean outline path."
    )
    p.add_argument("input_svg", type=Path, help="Input copper_top.svg")
    p.add_argument(
        "output_svg",
        type=Path,
        nargs="?",
        help="Output outline SVG (default: <input>_smooth.svg)",
    )
    p.add_argument(
        "--ground-svg",
        type=Path,
        default=None,
        help="Optional input copper_ground.svg to smooth and align with top",
    )
    p.add_argument(
        "--ground-output-svg",
        type=Path,
        default=None,
        help="Optional smoothed ground output (default: <ground>_smooth.svg)",
    )
    p.add_argument(
        "--aligned-output-svg",
        type=Path,
        default=None,
        help=(
            "Optional combined aligned top+ground SVG "
            "(default: <input>_smooth_aligned.svg when --ground-svg is set)"
        ),
    )
    p.add_argument(
        "--scale",
        type=float,
        default=24.0,
        help="Raster supersampling in pixels per SVG unit (default: 24)",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=1.2,
        help="Smoothing radius in SVG units (default: 1.2)",
    )
    p.add_argument(
        "--mode",
        choices=["close", "open", "close-open", "open-close", "gaussian"],
        default="gaussian",
        help="Smoothing operator (default: gaussian)",
    )
    p.add_argument(
        "--simplify",
        type=float,
        default=0.10,
        help="Contour simplify tolerance in SVG units (default: 0.10)",
    )
    p.add_argument(
        "--chaikin-iters",
        type=int,
        default=3,
        help="Chaikin corner-rounding iterations after contour extraction (default: 3)",
    )
    p.add_argument(
        "--min-island-area",
        type=float,
        default=0.0,
        help="Drop islands/holes smaller than this area in SVG units^2",
    )
    p.add_argument(
        "--debug-mask",
        type=Path,
        default=None,
        help="Optional PNG path to write the smoothed raster mask",
    )
    p.add_argument(
        "--debug-mask-ground",
        type=Path,
        default=None,
        help="Optional PNG path to write the smoothed ground raster mask",
    )
    p.add_argument(
        "--source-scale",
        type=float,
        default=5.0,
        help="Source exporter scale used in copper_top/ground SVG creation (default: 5)",
    )
    p.add_argument(
        "--source-margin",
        type=float,
        default=10.0,
        help="Source exporter margin used in copper_top/ground SVG creation (default: 10)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path: Path = args.input_svg
    out_path: Path = (
        args.output_svg if args.output_svg else in_path.with_name(f"{in_path.stem}_smooth.svg")
    )

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if args.radius < 0:
        raise ValueError("--radius must be >= 0")

    top = _process_layer(
        in_path=in_path,
        scale=args.scale,
        radius=args.radius,
        mode=args.mode,
        simplify=args.simplify,
        min_island_area=args.min_island_area,
        chaikin_iters=args.chaikin_iters,
        debug_mask_path=args.debug_mask,
    )
    _write_svg(
        out_path,
        width_attr=top["width_attr"],
        height_attr=top["height_attr"],
        vb=top["vb"],
        path_d=top["path_d"],
    )

    print(f"Input polygons: {len(top['polygons_local'])}")
    print(
        f"Mask size: {top['mask_shape'][1]}x{top['mask_shape'][0]} px "
        f"(scale={args.scale})"
    )
    print(
        f"Smoothing: mode={args.mode}, radius={args.radius} units "
        f"({top['radius_px']} px)"
    )
    if top["grid_step"] is not None:
        print(f"Estimated tile step: {top['grid_step']:.4f} units")
        if args.radius < 0.5 * top["grid_step"]:
            print(
                "Warning: radius is small relative to tile step; try --radius "
                f"{0.75 * top['grid_step']:.2f} to {1.25 * top['grid_step']:.2f}."
            )
    print(f"Chaikin iterations: {args.chaikin_iters}")
    print(f"Wrote: {out_path}")

    if args.ground_svg is not None:
        ground_in = args.ground_svg
        ground_out = (
            args.ground_output_svg
            if args.ground_output_svg is not None
            else ground_in.with_name(f"{ground_in.stem}_smooth.svg")
        )
        aligned_out = (
            args.aligned_output_svg
            if args.aligned_output_svg is not None
            else out_path.with_name(f"{out_path.stem}_aligned.svg")
        )
        ground = _process_layer(
            in_path=ground_in,
            scale=args.scale,
            radius=args.radius,
            mode=args.mode,
            simplify=args.simplify,
            min_island_area=args.min_island_area,
            chaikin_iters=args.chaikin_iters,
            debug_mask_path=args.debug_mask_ground,
        )
        _write_svg(
            ground_out,
            width_attr=ground["width_attr"],
            height_attr=ground["height_attr"],
            vb=ground["vb"],
            path_d=ground["path_d"],
        )

        cfg = _load_model_cfg_near([in_path, ground_in])
        if cfg is None:
            raise RuntimeError(
                "Could not locate model_cfg in nearby params.json for layer alignment. "
                "Place top/ground SVGs next to openEMS run params.json."
            )
        tx, ty, meta = _compute_ground_alignment_translation(
            cfg=cfg,
            top_polys_local=top["polygons_local"],
            source_scale=args.source_scale,
            source_margin=args.source_margin,
        )
        _write_aligned_svg(
            out_path=aligned_out,
            top_d=top["path_d"],
            top_vb=top["vb"],
            ground_d=ground["path_d"],
            ground_vb=ground["vb"],
            ground_translate=(tx, ty),
        )
        print(f"Ground polygons: {len(ground['polygons_local'])}")
        print(f"Wrote: {ground_out}")
        print(f"Wrote: {aligned_out}")
        print(
            "Alignment (ground->top frame) translate: "
            f"dx={tx:.4f}, dy={ty:.4f} (source_scale={args.source_scale}, "
            f"source_margin={args.source_margin})"
        )
        print(
            "Alignment anchors: "
            f"feed_offset_mm={meta['feed_offset_mm']:.4f}, "
            f"ground_min_x_mm={meta['ground_min_x_mm']:.4f}, "
            f"wedge_height_mm={meta['wedge_height_mm']:.4f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

"""Export top copper geometry to SVG."""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np

from sim import common
from sim.models import MODEL_CONFIGS


def _load_run(run_dir: str) -> Tuple[str, dict, np.ndarray, float]:
    params_path = os.path.join(run_dir, "params.json")
    geom_path = os.path.join(run_dir, "geometry.npz")
    if not os.path.exists(params_path) or not os.path.exists(geom_path):
        raise FileNotFoundError("Missing params.json or geometry.npz in run directory")
    with open(params_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)
    data = np.load(geom_path)
    design_indices = data["design_indices"]
    resolution_mm = float(data["resolution_mm"])
    return params["model"], params, design_indices, resolution_mm


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
                rects.append((x0, y0, width, height))
                run_start = None
    return rects


def export_svg(run_dir: str, out_path: str, threshold: float, include_feed: bool) -> None:
    model_name, params, design_indices, resolution_mm = _load_run(run_dir)
    config = MODEL_CONFIGS[model_name]

    points = common.wedge_polygon_points(
        config.inner_radius_mm,
        config.outer_radius_mm,
        config.slice_angle_deg,
        n=48,
    )
    shifted, _ = common.normalize_polygon(points)
    min_x, max_x, min_y, max_y = common.polygon_bounds(shifted)
    wedge_w = max_x - min_x
    wedge_h = max_y - min_y

    device_origin_x = config.feed_offset_mm + config.feed_length_mm
    device_origin_y = (wedge_h - config.design_width_mm) * 0.5

    mask = (design_indices[..., 0] >= threshold).astype(np.uint8)
    rects = _rects_from_mask(mask, resolution_mm, device_origin_x, device_origin_y)

    feed_rect = None
    if include_feed:
        feed_x = config.feed_offset_mm
        feed_y = (wedge_h - config.feed_width_mm) * 0.5
        feed_rect = (feed_x, feed_y, config.feed_length_mm, config.feed_width_mm)

    width = wedge_w
    height = wedge_h

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        handle.write(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}mm\" height=\"{height}mm\" "
            f"viewBox=\"0 0 {width} {height}\">\n"
        )
        handle.write(f"<g transform=\"translate(0 {height}) scale(1 -1)\">\n")
        handle.write("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n")
        for x0, y0, w, h in rects:
            handle.write(
                f"<rect x=\"{x0:.3f}\" y=\"{y0:.3f}\" width=\"{w:.3f}\" height=\"{h:.3f}\" "
                "fill=\"#c47f2c\" stroke=\"none\"/>\n"
            )
        if feed_rect:
            x0, y0, w, h = feed_rect
            handle.write(
                f"<rect x=\"{x0:.3f}\" y=\"{y0:.3f}\" width=\"{w:.3f}\" height=\"{h:.3f}\" "
                "fill=\"#c47f2c\" stroke=\"none\"/>\n"
            )
        handle.write("</g>\n</svg>\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory with geometry.npz")
    parser.add_argument("--out", default=None, help="Output SVG path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-feed", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run)
    out_path = args.out or os.path.join(run_dir, "top_copper.svg")
    export_svg(run_dir, out_path, args.threshold, args.include_feed)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

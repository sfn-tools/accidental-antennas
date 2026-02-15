"""Export top copper polygons to SVG for KiCad redraw."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from sim import model_common as mc
from sim import model_dir24_quasiyagi
from sim import model_dir24_acs_monopole_yagi
from sim import model_dir24_acs_ursi_uniplanar
from sim import model_dir24_acs_ursi_ground
from sim import model_dir5_quasiyagi
from sim import model_dir5_ms_cps_dipole_yagi
from sim import model_omni_dual_ifa


MODEL_MAP = {
    "dir24": model_dir24_quasiyagi,
    "dir24_acs": model_dir24_acs_monopole_yagi,
    "dir24_acs_ursi_uni": model_dir24_acs_ursi_uniplanar,
    "dir24_acs_ursi_gnd": model_dir24_acs_ursi_ground,
    "dir5": model_dir5_quasiyagi,
    "dir5_ms_cps": model_dir5_ms_cps_dipole_yagi,
    "omni": model_omni_dual_ifa,
}


def _load_best(root: str, model_name: str) -> Dict:
    specific = os.path.join(root, "designs", f"best_{model_name}.json")
    fallback = os.path.join(root, "designs", "best.json")
    path = specific if os.path.exists(specific) else fallback
    payload = mc.load_json(path)
    params = payload.get("params", payload)
    return {"params": params, "path": path}


def _svg_points(points, min_x, max_y, scale, margin):
    coords = []
    for x, y in points:
        sx = (x - min_x) * scale + margin
        sy = (max_y - y) * scale + margin
        coords.append(f"{sx:.3f},{sy:.3f}")
    return " ".join(coords)


def write_svg(path: str, polys: List[mc.Polygon2D]) -> Dict:
    if not polys:
        return {}
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

    return {"min_x_mm": min_x, "max_x_mm": max_x, "min_y_mm": min_y, "max_y_mm": max_y}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default=None)
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_constraints = mc.load_constraints(os.path.join(root, "designs", "constraints.json"))
    export_root = os.path.join(root, "exports")
    os.makedirs(export_root, exist_ok=True)

    models = MODEL_MAP.items()
    if args.model:
        models = [(args.model, MODEL_MAP[args.model])]

    for model_name, model in models:
        constraints = mc.apply_constraint_overrides(
            base_constraints, getattr(model, "DEFAULT_CONSTRAINTS", None)
        )
        best = _load_best(root, model_name)
        merged = dict(model.default_params(constraints))
        merged.update(best["params"])
        geom, _ = model.build_geometry(merged, constraints)
        svg_path = os.path.join(export_root, f"{model_name}_top.svg")
        bounds_top = write_svg(svg_path, geom.top_polys)

        bounds_ground = {}
        if geom.ground_polys:
            ground_path = os.path.join(export_root, f"{model_name}_ground.svg")
            bounds_ground = write_svg(ground_path, geom.ground_polys)

        dims = {
            "model": model_name,
            "source": best["path"],
            "params": merged,
            "bounds_top": bounds_top,
            "bounds_ground": bounds_ground,
        }
        with open(os.path.join(export_root, f"{model_name}_dims.json"), "w", encoding="utf-8") as handle:
            json.dump(dims, handle, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

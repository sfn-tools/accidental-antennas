"""Common utilities for openEMS antenna models."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


UNIT = 1e-3  # mm


@dataclass
class SubstrateConfig:
    eps_r: float = 4.3
    tan_delta: float = 0.02
    thickness_mm: float = 1.6
    copper_thickness_mm: float = 0.035


@dataclass
class SliceConstraints:
    slice_angle_deg: float = 30.0
    outer_radius_mm: float = 80.0
    inner_radius_mm: float = 20.0
    keepout_edge_mm: float = 2.0
    feed_location_mm: float = 24.0
    min_trace_mm: float = 0.15
    min_gap_mm: float = 0.15


@dataclass
class QualitySettings:
    name: str
    max_cell_mm: float
    end_criteria: float
    air_margin_mm: float
    smooth_ratio: float = 1.4
    nr_ts: int = 20000
    snap_mm: float = 0.1


@dataclass
class Polygon2D:
    name: str
    points: List[Tuple[float, float]]


@dataclass
class Geometry:
    top_polys: List[Polygon2D] = field(default_factory=list)
    ground_polys: List[Polygon2D] = field(default_factory=list)
    feed_point: Tuple[float, float] = (0.0, 0.0)
    meta: Dict[str, float] = field(default_factory=dict)

    def all_points(self) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for poly in self.top_polys:
            pts.extend(poly.points)
        for poly in self.ground_polys:
            pts.extend(poly.points)
        return pts


class OpenEMSImportError(RuntimeError):
    pass


def require_openems():
    try:
        import openEMS  # noqa: F401
        import CSXCAD  # noqa: F401
    except Exception as exc:
        msg = (
            "openEMS Python modules not available.\n"
            "Activate your openEMS environment first (for example):\n"
            "  source ../scripts/env.sh\n"
            "Then rerun using that Python interpreter."
        )
        raise OpenEMSImportError(msg) from exc


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_constraints(path: str) -> SliceConstraints:
    data = load_json(path)
    return SliceConstraints(**data)


def apply_constraint_overrides(
    constraints: SliceConstraints, overrides: Dict[str, float] | None
) -> SliceConstraints:
    if not overrides:
        return constraints
    merged = dict(constraints.__dict__)
    for key, val in overrides.items():
        if key in merged:
            merged[key] = val
    return SliceConstraints(**merged)


def load_substrate(params: Dict | None = None, base: SubstrateConfig | Dict | None = None) -> SubstrateConfig:
    if isinstance(base, SubstrateConfig):
        merged = SubstrateConfig(**base.__dict__)
    else:
        merged = SubstrateConfig()
        if isinstance(base, dict):
            for key, val in base.items():
                if hasattr(merged, key):
                    setattr(merged, key, val)
    if params:
        for key, val in params.items():
            if hasattr(merged, key):
                setattr(merged, key, val)
    return merged


def quality_from_name(name: str, fmax_hz: float, eps_r: float, min_feature_mm: float) -> QualitySettings:
    name = name.lower()
    c0 = 299792458.0
    lambda_min_mm = c0 / fmax_hz * 1e3
    if name == "fast":
        cells = 10
        end_criteria = 1e-3
        air_margin = 0.25 * lambda_min_mm
        nr_ts = 30000
        snap_mm = min(0.1, max(min_feature_mm * 0.5, 0.05))
    elif name == "high":
        cells = 20
        end_criteria = 5e-5
        air_margin = 0.45 * lambda_min_mm
        nr_ts = 80000
        snap_mm = 0.05
    else:
        name = "medium"
        cells = 15
        end_criteria = 2e-4
        air_margin = 0.35 * lambda_min_mm
        nr_ts = 40000
        snap_mm = 0.1

    max_cell = lambda_min_mm / (cells * math.sqrt(max(eps_r, 1.0)))
    if name == "fast":
        max_cell = max(max_cell, 0.4)
    elif name == "medium":
        max_cell = max(max_cell, max(min_feature_mm * 0.8, 0.2))
    else:
        max_cell = max(max_cell, max(min_feature_mm * 0.5, 0.1))
    return QualitySettings(
        name=name,
        max_cell_mm=max_cell,
        end_criteria=end_criteria,
        air_margin_mm=air_margin,
        nr_ts=nr_ts,
        snap_mm=snap_mm,
    )


def calc_kappa(eps_r: float, tan_delta: float, f0_hz: float) -> float:
    from openEMS import physical_constants as pc

    return 2 * math.pi * f0_hz * pc.EPS0 * eps_r * tan_delta


def wedge_polygon(inner_r: float, outer_r: float, angle_deg: float, n: int = 24) -> Polygon2D:
    half = math.radians(angle_deg / 2.0)
    angles_outer = [(-half) + i * (2 * half) / (n - 1) for i in range(n)]
    angles_inner = list(reversed(angles_outer))
    outer = [(outer_r * math.cos(a), outer_r * math.sin(a)) for a in angles_outer]
    inner = [(inner_r * math.cos(a), inner_r * math.sin(a)) for a in angles_inner]
    points = outer + inner
    return Polygon2D(name="ground", points=points)


def rect_polygon(name: str, x0: float, x1: float, y0: float, y1: float) -> Polygon2D:
    return Polygon2D(
        name=name,
        points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
    )


def rotate_points(points: Iterable[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    ang = math.radians(angle_deg)
    c = math.cos(ang)
    s = math.sin(ang)
    out = []
    for x, y in points:
        out.append((x * c - y * s, x * s + y * c))
    return out


def rotate_geometry(geom: Geometry, angle_deg: float) -> Geometry:
    rotated = Geometry(feed_point=geom.feed_point, meta=dict(geom.meta))
    rotated.top_polys = [
        Polygon2D(name=poly.name, points=rotate_points(poly.points, angle_deg))
        for poly in geom.top_polys
    ]
    rotated.ground_polys = [
        Polygon2D(name=poly.name, points=rotate_points(poly.points, angle_deg))
        for poly in geom.ground_polys
    ]
    rotated.feed_point = rotate_points([geom.feed_point], angle_deg)[0]
    if "port_defs" in geom.meta:
        port_defs = []
        for port in geom.meta["port_defs"]:
            start = rotate_points([tuple(port["start"][:2])], angle_deg)[0]
            stop = rotate_points([tuple(port["stop"][:2])], angle_deg)[0]
            rotated_port = {
                "start": [start[0], start[1], port["start"][2]],
                "stop": [stop[0], stop[1], port["stop"][2]],
                "exc_dir": port.get("exc_dir", "z"),
                "R": port.get("R", 50),
                "port_type": port.get("port_type", "lumped"),
                "prop_dir": port.get("prop_dir", "x"),
            }
            if "feed_shift" in port:
                rotated_port["feed_shift"] = port["feed_shift"]
            if "measplane_shift" in port:
                rotated_port["measplane_shift"] = port["measplane_shift"]
            if "priority" in port:
                rotated_port["priority"] = port["priority"]
            port_defs.append(rotated_port)
        rotated.meta["port_defs"] = port_defs
    return rotated


def geometry_bounds(geom: Geometry) -> Tuple[float, float, float, float]:
    pts = geom.all_points()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def geometry_edges(geom: Geometry) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for poly in geom.top_polys:
        for x, y in poly.points:
            xs.append(x)
            ys.append(y)
    for poly in geom.ground_polys:
        px = [p[0] for p in poly.points]
        py = [p[1] for p in poly.points]
        xs.extend([min(px), max(px)])
        ys.extend([min(py), max(py)])
    xs.append(geom.feed_point[0])
    ys.append(geom.feed_point[1])
    for port in geom.meta.get("port_defs", []):
        xs.append(port["start"][0])
        xs.append(port["stop"][0])
        ys.append(port["start"][1])
        ys.append(port["stop"][1])
    return xs, ys


def wedge_violation_penalty(geom: Geometry, constraints: SliceConstraints) -> float:
    penalty = 0.0
    half = constraints.slice_angle_deg / 2.0
    for x, y in geom.all_points():
        r = math.hypot(x, y)
        if r < constraints.inner_radius_mm + constraints.keepout_edge_mm:
            penalty += (constraints.inner_radius_mm + constraints.keepout_edge_mm - r)
        if r > constraints.outer_radius_mm - constraints.keepout_edge_mm:
            penalty += (r - (constraints.outer_radius_mm - constraints.keepout_edge_mm))
        ang = math.degrees(math.atan2(y, x))
        if abs(ang) > half:
            penalty += (abs(ang) - half) * 0.5
    return penalty


def hash_params(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def build_mesh_lines(lines: List[float], max_cell_mm: float, ratio: float) -> List[float]:
    from CSXCAD.SmoothMeshLines import SmoothMeshLines

    uniq = sorted(set(lines))
    return SmoothMeshLines(uniq, max_cell_mm, ratio)


def add_polygon_property(prop, poly: Polygon2D, z_mm: float) -> None:
    xs = [p[0] for p in poly.points]
    ys = [p[1] for p in poly.points]
    prop.AddPolygon([xs, ys], norm_dir="z", elevation=z_mm)


def _snap_poly(poly: Polygon2D, snap_mm: float) -> Polygon2D:
    if snap_mm <= 0:
        return poly
    snapped = []
    for x, y in poly.points:
        sx = round(x / snap_mm) * snap_mm
        sy = round(y / snap_mm) * snap_mm
        snapped.append((sx, sy))
    return Polygon2D(name=poly.name, points=snapped)


def build_simulation(
    geom: Geometry,
    substrate: SubstrateConfig,
    constraints: SliceConstraints,
    quality: QualitySettings,
    f0_hz: float,
    fc_hz: float,
    fmax_hz: float,
    excite_port: int = 0,
    port_count: int = 1,
) -> Tuple[object, List[object], object, Dict]:
    require_openems()
    import numpy as np
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS import ports

    FDTD = openEMS(EndCriteria=quality.end_criteria, NrTS=quality.nr_ts)
    FDTD.SetGaussExcite(f0_hz, fc_hz)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(UNIT)

    x_min, x_max, y_min, y_max = geometry_bounds(geom)
    x_min -= quality.air_margin_mm
    x_max += quality.air_margin_mm
    y_min -= quality.air_margin_mm
    y_max += quality.air_margin_mm
    z_min = -quality.air_margin_mm * 0.7
    z_max = substrate.thickness_mm + quality.air_margin_mm

    x_edges, y_edges = geometry_edges(geom)
    if quality.snap_mm > 0:
        x_edges = [round(x / quality.snap_mm) * quality.snap_mm for x in x_edges]
        y_edges = [round(y / quality.snap_mm) * quality.snap_mm for y in y_edges]
    x_lines = build_mesh_lines(x_edges + [x_min, x_max], quality.max_cell_mm, quality.smooth_ratio)
    y_lines = build_mesh_lines(y_edges + [y_min, y_max], quality.max_cell_mm, quality.smooth_ratio)
    z_lines = build_mesh_lines([z_min, 0.0, substrate.thickness_mm, z_max], quality.max_cell_mm, quality.smooth_ratio)

    grid.SetLines("x", np.array(x_lines))
    grid.SetLines("y", np.array(y_lines))
    grid.SetLines("z", np.array(z_lines))

    # materials
    kappa = calc_kappa(substrate.eps_r, substrate.tan_delta, f0_hz)
    sub = CSX.AddMaterial("FR4")
    sub.SetMaterialProperty(epsilon=substrate.eps_r, kappa=kappa)
    sub.AddBox([x_min, y_min, 0.0], [x_max, y_max, substrate.thickness_mm])

    gnd = CSX.AddMetal("gnd")
    for poly in geom.ground_polys:
        add_polygon_property(gnd, _snap_poly(poly, quality.snap_mm), 0.0)

    top = CSX.AddMetal("top")
    for poly in geom.top_polys:
        add_polygon_property(top, _snap_poly(poly, quality.snap_mm), substrate.thickness_mm)

    ports_out: List[object] = []
    port_defs = geom.meta.get("port_defs")
    if port_defs:
        port_count = len(port_defs)
        excite_port = min(excite_port, port_count - 1)
        for idx, port_def in enumerate(port_defs):
            excite = 1 if idx == excite_port else 0
            start = list(port_def["start"])
            stop = list(port_def["stop"])
            if start[2] is None:
                start[2] = substrate.thickness_mm
            if stop[2] is None:
                stop[2] = substrate.thickness_mm
            if quality.snap_mm > 0:
                start[0] = round(start[0] / quality.snap_mm) * quality.snap_mm
                start[1] = round(start[1] / quality.snap_mm) * quality.snap_mm
                stop[0] = round(stop[0] / quality.snap_mm) * quality.snap_mm
                stop[1] = round(stop[1] / quality.snap_mm) * quality.snap_mm
            port_type = port_def.get("port_type", "lumped")
            exc_dir = port_def.get("exc_dir", "z")
            if port_type == "msl":
                port_kwargs = {}
                if "feed_shift" in port_def:
                    port_kwargs["FeedShift"] = port_def["feed_shift"]
                if "measplane_shift" in port_def:
                    port_kwargs["MeasPlaneShift"] = port_def["measplane_shift"]
                if "priority" in port_def:
                    port_kwargs["priority"] = port_def["priority"]
                port = ports.MSLPort(
                    CSX,
                    port_nr=idx + 1,
                    metal_prop=top,
                    start=start,
                    stop=stop,
                    prop_dir=port_def.get("prop_dir", "x"),
                    exc_dir=exc_dir,
                    excite=excite,
                    **port_kwargs,
                )
            else:
                port = ports.LumpedPort(
                    CSX,
                    port_nr=idx + 1,
                    R=port_def.get("R", 50),
                    start=start,
                    stop=stop,
                    exc_dir=exc_dir,
                    excite=excite,
                )
            ports_out.append(port)
    else:
        for idx in range(port_count):
            excite = 1 if idx == excite_port else 0
            px, py = geom.feed_point if port_count == 1 else geom.meta["feed_points"][idx]
            if quality.snap_mm > 0:
                px = round(px / quality.snap_mm) * quality.snap_mm
                py = round(py / quality.snap_mm) * quality.snap_mm
            port = ports.LumpedPort(
                CSX,
                port_nr=idx + 1,
                R=50,
                start=[px, py, 0.0],
                stop=[px, py, substrate.thickness_mm],
                exc_dir="z",
                excite=excite,
            )
            ports_out.append(port)

    FDTD.SetCSX(CSX)

    # nf2ff box: 3 cells away from boundaries
    start = [x_lines[3], y_lines[3], z_lines[3]]
    stop = [x_lines[-4], y_lines[-4], z_lines[-4]]
    nf2ff = FDTD.CreateNF2FFBox(name="nf2ff", start=start, stop=stop)

    min_cell = min(
        float(np.min(np.diff(x_lines))),
        float(np.min(np.diff(y_lines))),
        float(np.min(np.diff(z_lines))),
    )
    meta = {
        "nf2ff_start_mm": start,
        "nf2ff_stop_mm": stop,
        "mesh_max_cell_mm": quality.max_cell_mm,
        "mesh_min_cell_mm": min_cell,
    }

    return FDTD, ports_out, nf2ff, meta

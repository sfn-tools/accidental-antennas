"""Shared helpers for FDTDX antenna simulations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class QualitySettings:
    name: str
    resolution_mm: float
    time_cycles: float


QUALITY_PRESETS = {
    "coarse": QualitySettings("coarse", resolution_mm=2.5, time_cycles=6.0),
    "mid": QualitySettings("mid", resolution_mm=1.5, time_cycles=8.0),
    "medium": QualitySettings("medium", resolution_mm=1.0, time_cycles=10.0),
    "fine": QualitySettings("fine", resolution_mm=0.5, time_cycles=14.0),
    "fast": QualitySettings("fast", resolution_mm=2.5, time_cycles=6.0),
    "high": QualitySettings("high", resolution_mm=0.5, time_cycles=14.0),
}


def resolve_quality(name: str, base_resolution_mm: float) -> QualitySettings:
    preset = QUALITY_PRESETS.get(name, QUALITY_PRESETS["medium"])
    return QualitySettings(
        name=preset.name,
        resolution_mm=base_resolution_mm * preset.resolution_mm,
        time_cycles=preset.time_cycles,
    )


def hash_payload(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def wedge_polygon_points(
    inner_r_mm: float,
    outer_r_mm: float,
    angle_deg: float,
    n: int = 36,
) -> List[Tuple[float, float]]:
    half = math.radians(angle_deg / 2.0)
    angles_outer = [(-half) + i * (2 * half) / (n - 1) for i in range(n)]
    angles_inner = list(reversed(angles_outer))
    outer = [(outer_r_mm * math.cos(a), outer_r_mm * math.sin(a)) for a in angles_outer]
    inner = [(inner_r_mm * math.cos(a), inner_r_mm * math.sin(a)) for a in angles_inner]
    points = outer + inner
    return points


def normalize_polygon(points: Iterable[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, min_y = min(xs), min(ys)
    shifted = [(x - min_x, y - min_y) for x, y in pts]
    return shifted, (min_x, min_y)


def polygon_bounds(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def mm_to_m(val_mm: float) -> float:
    return val_mm * 1e-3


def m_to_mm(val_m: float) -> float:
    return val_m * 1e3


def validate_mesh_constraints(
    resolution_mm: float,
    min_trace_mm: float,
    min_gap_mm: float,
    feed_width_mm: float,
    gap_mm: float,
) -> tuple[bool, str]:
    if feed_width_mm < min_trace_mm:
        return False, f"feed_width_mm {feed_width_mm:.3f} < min_trace_mm {min_trace_mm:.3f}"
    if feed_width_mm < 6.0 * resolution_mm:
        return False, f"feed_width_mm {feed_width_mm:.3f} < 6*dx {6.0*resolution_mm:.3f}"
    if gap_mm < min_gap_mm:
        return False, f"gap_mm {gap_mm:.3f} < min_gap_mm {min_gap_mm:.3f}"
    if gap_mm < 4.0 * resolution_mm:
        return False, f"gap_mm {gap_mm:.3f} < 4*dx {4.0*resolution_mm:.3f}"
    return True, "ok"

"""Shared defaults for antenna model configurations."""

from __future__ import annotations


COMMON_DEFAULTS = {
    "substrate_thickness_mm": 1.6,
    "copper_thickness_mm": 0.0348,
    "min_trace_mm": 0.2,
    "min_gap_mm": 0.2,
    "eta_min": 0.2,
    "projection_beta": 6.0,
}

S15_WEDGE_DEFAULTS = {
    "slice_angle_deg": 24.0,
    "inner_radius_mm": 20.0,
    "outer_radius_mm": 75.0,
}

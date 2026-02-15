"""Scoring utilities for optimization."""

from __future__ import annotations

from typing import Dict


def _penalty_rl(rl_min_db: float, target_db: float = 10.0, scale: float = 1000.0) -> float:
    if rl_min_db is None:
        return scale * target_db
    return max(0.0, target_db - rl_min_db) * scale


def _penalty_iso(iso_db: float, threshold_db: float = -15.0) -> float:
    return max(0.0, iso_db - threshold_db)


def _resonance_penalty(metrics: Dict, weight: float) -> float:
    pen_f = metrics.get("pen_f", 0.0)
    if pen_f is None:
        return weight
    return float(pen_f) * weight


def score_directional(metrics: Dict, stage: str) -> float:
    if not metrics.get("valid", True):
        return 1e6 + metrics.get("constraint_penalty", 0.0) * 1e4
    rl_min = metrics.get("rl_min_in_band_db", metrics.get("rl_min_db", 0.0))
    iso = metrics.get("iso_db", -200.0)
    constraint_penalty = metrics.get("constraint_penalty", 0.0)

    if stage == "lock":
        rl_peak = metrics.get("rl_peak_db", 0.0)
        penalty = _resonance_penalty(metrics, 5000.0)
        penalty += _penalty_rl(rl_peak, target_db=10.0, scale=50.0)
        penalty += 5.0 * constraint_penalty
        return penalty

    if rl_min is None or rl_min < 8.0:
        return 1e5 + constraint_penalty * 1e4

    gain = metrics.get("gain_fwd_db", -200.0)
    fb = metrics.get("fb_db", 0.0)
    penalty = _resonance_penalty(metrics, 50.0)
    penalty += _penalty_rl(rl_min) + 1.5 * _penalty_iso(iso) + 5.0 * constraint_penalty
    return -(gain + fb) + penalty


def score_omni(metrics: Dict) -> float:
    if not metrics.get("valid", True):
        return 1e6 + metrics.get("constraint_penalty", 0.0) * 1e4
    rl_24 = metrics.get("rl_min_24_db", 0.0)
    rl_5 = metrics.get("rl_min_5_db", 0.0)
    ripple_24 = metrics.get("ripple_24_db", 100.0)
    ripple_5 = metrics.get("ripple_5_db", 100.0)
    constraint_penalty = metrics.get("constraint_penalty", 0.0)

    penalty = _penalty_rl(rl_24) + _penalty_rl(rl_5)
    penalty += 0.5 * (ripple_24 + ripple_5) + 5.0 * constraint_penalty
    return penalty


def score(metrics: Dict, model_name: str, stage: str = "final") -> float:
    if model_name.startswith("omni"):
        return score_omni(metrics)
    return score_directional(metrics, stage)

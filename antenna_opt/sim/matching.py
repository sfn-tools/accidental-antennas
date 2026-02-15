"""L-match helpers for openEMS post-processing."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def _range_penalty(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1e6
    if value < lo:
        return ((lo / value) - 1.0) ** 2
    if value > hi:
        return ((value / hi) - 1.0) ** 2
    return 0.0


def _component_from_reactance(x_ohm: float, f0_hz: float) -> Tuple[str, float]:
    if not np.isfinite(x_ohm) or abs(x_ohm) < 1e-18:
        return ("", float("nan"))
    w0 = 2.0 * math.pi * f0_hz
    if x_ohm > 0.0:
        return ("L", x_ohm / w0)
    return ("C", -1.0 / (w0 * x_ohm))


def _reactance_from_component(x0: float, f0_hz: float, freq_hz: np.ndarray) -> np.ndarray:
    freq = np.maximum(freq_hz, 1e-6)
    if x0 >= 0.0:
        return x0 * (freq / f0_hz)
    return x0 * (f0_hz / freq)


def _series_then_shunt(R: float, X: float, z0: float) -> list[Tuple[float, float]]:
    rad = R * (z0 - R)
    if R <= 0.0 or rad <= 0.0:
        return []
    root = math.sqrt(rad)
    candidates = []
    for xs in (-X + root, -X - root):
        x_tot = X + xs
        denom = R * R + x_tot * x_tot
        if denom <= 0.0:
            continue
        bp = x_tot / denom
        if abs(bp) < 1e-18:
            continue
        xp = -1.0 / bp
        candidates.append((xs, xp))
    return candidates


def _shunt_then_series(R: float, X: float, z0: float) -> list[Tuple[float, float]]:
    denom = R * R + X * X
    if R <= 0.0 or denom <= 0.0:
        return []
    G = R / denom
    B = -X / denom
    rad = G / z0 - G * G
    if rad <= 0.0:
        return []
    root = math.sqrt(rad)
    candidates = []
    for bp in (-B + root, -B - root):
        if abs(bp) < 1e-18:
            continue
        denom2 = G * G + (B + bp) * (B + bp)
        if denom2 <= 0.0:
            continue
        R1 = G / denom2
        X1 = -(B + bp) / denom2
        xs = -X1
        xp = -1.0 / bp
        candidates.append((xs, xp))
    return candidates


def calc_l_match(
    zin_f0: complex,
    f0_hz: float,
    z0: float = 50.0,
    l_range_nh: Tuple[float, float] = (0.2, 20.0),
    c_range_pf: Tuple[float, float] = (0.1, 10.0),
) -> Optional[Dict[str, float]]:
    if not np.isfinite(zin_f0):
        return None
    R = float(np.real(zin_f0))
    X = float(np.imag(zin_f0))
    if not np.isfinite(R) or not np.isfinite(X) or R <= 0.0:
        return None

    l_min_h = l_range_nh[0] * 1e-9
    l_max_h = l_range_nh[1] * 1e-9
    c_min_f = c_range_pf[0] * 1e-12
    c_max_f = c_range_pf[1] * 1e-12

    candidates = []
    for xs, xp in _series_then_shunt(R, X, z0):
        candidates.append(("series-shunt", xs, xp))
    for xs, xp in _shunt_then_series(R, X, z0):
        candidates.append(("shunt-series", xs, xp))
    if not candidates:
        return None

    best = None
    for topology, xs, xp in candidates:
        series_type, series_val = _component_from_reactance(xs, f0_hz)
        shunt_type, shunt_val = _component_from_reactance(xp, f0_hz)
        if not series_type or not shunt_type:
            continue
        series_pen = _range_penalty(series_val, l_min_h if series_type == "L" else c_min_f, l_max_h if series_type == "L" else c_max_f)
        shunt_pen = _range_penalty(shunt_val, l_min_h if shunt_type == "L" else c_min_f, l_max_h if shunt_type == "L" else c_max_f)
        penalty = series_pen + shunt_pen
        entry = {
            "topology": topology,
            "series_react_ohm": xs,
            "shunt_react_ohm": xp,
            "series_type": series_type,
            "series_value": series_val,
            "shunt_type": shunt_type,
            "shunt_value": shunt_val,
            "penalty": penalty,
            "f0_hz": f0_hz,
        }
        if best is None or entry["penalty"] < best["penalty"]:
            best = entry
    return best


def apply_l_match(
    zin: np.ndarray,
    freq_hz: np.ndarray,
    match: Dict[str, float],
    z0: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs0 = float(match["series_react_ohm"])
    xp0 = float(match["shunt_react_ohm"])
    f0_hz = float(match.get("f0_hz", freq_hz[0]))
    x_series = _reactance_from_component(xs0, f0_hz, freq_hz)
    x_shunt = _reactance_from_component(xp0, f0_hz, freq_hz)
    z_series = 1j * x_series
    z_shunt = 1j * x_shunt

    if match["topology"] == "series-shunt":
        z1 = zin + z_series
        y_in = 1.0 / z1 + 1.0 / z_shunt
        zin_matched = 1.0 / y_in
    else:
        y1 = 1.0 / zin + 1.0 / z_shunt
        z1 = 1.0 / y1
        zin_matched = z1 + z_series

    s11 = (zin_matched - z0) / (zin_matched + z0)
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-12)
    return zin_matched, s11, s11_db


def _reactance_candidates(
    f0_hz: float,
    l_range_nh: Tuple[float, float],
    c_range_pf: Tuple[float, float],
    samples: int,
) -> list[Dict[str, float]]:
    w0 = 2.0 * math.pi * f0_hz
    l_vals = np.logspace(math.log10(l_range_nh[0]), math.log10(l_range_nh[1]), samples)
    c_vals = np.logspace(math.log10(c_range_pf[0]), math.log10(c_range_pf[1]), samples)
    candidates: list[Dict[str, float]] = []
    for l_val in l_vals:
        candidates.append({"type": "L", "value": float(l_val) * 1e-9, "x0": float(w0 * l_val * 1e-9)})
    for c_val in c_vals:
        candidates.append({"type": "C", "value": float(c_val) * 1e-12, "x0": float(-1.0 / (w0 * c_val * 1e-12))})
    return candidates


def _apply_pi_match(
    zin: np.ndarray,
    freq_hz: np.ndarray,
    x_in: float,
    x_series: float,
    x_out: float,
    f0_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_in_arr = _reactance_from_component(x_in, f0_hz, freq_hz)
    x_series_arr = _reactance_from_component(x_series, f0_hz, freq_hz)
    x_out_arr = _reactance_from_component(x_out, f0_hz, freq_hz)
    z_in = 1j * x_in_arr
    z_series = 1j * x_series_arr
    z_out = 1j * x_out_arr

    z1 = 1.0 / (1.0 / zin + 1.0 / (z_out + 1e-18))
    z2 = z1 + z_series
    z_matched = 1.0 / (1.0 / z2 + 1.0 / (z_in + 1e-18))
    s11 = (z_matched - 50.0) / (z_matched + 50.0)
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-12)
    return z_matched, s11, s11_db


def calc_pi_match(
    zin: np.ndarray,
    freq_hz: np.ndarray,
    f0_hz: float,
    f_low: float,
    f_high: float,
    z0: float = 50.0,
    l_range_nh: Tuple[float, float] = (0.5, 20.0),
    c_range_pf: Tuple[float, float] = (0.2, 10.0),
    samples: int = 8,
) -> Optional[Dict[str, float]]:
    if freq_hz.size == 0:
        return None
    band_mask = (freq_hz >= f_low) & (freq_hz <= f_high)
    if not np.any(band_mask):
        return None

    candidates = _reactance_candidates(f0_hz, l_range_nh, c_range_pf, samples)
    if not candidates:
        return None

    best: Optional[Dict[str, float]] = None
    best_rl = -1e9
    best_bw = 0.0
    best_s11_db = None
    best_s11 = None
    best_zin = None
    best_f_peak = None
    best_rl_peak = None

    for cin in candidates:
        for cser in candidates:
            for cout in candidates:
                z_m, s11, s11_db = _apply_pi_match(
                    zin, freq_hz, cin["x0"], cser["x0"], cout["x0"], f0_hz
                )
                rl_db = -s11_db
                rl_band = rl_db[band_mask]
                rl_min = float(np.min(rl_band))
                if rl_min < best_rl:
                    continue
                bw_hz, bw_frac = matched_bandwidth(freq_hz, rl_db, f_low, f_high, rl_target=10.0)
                if rl_min > best_rl or bw_frac > best_bw:
                    best_rl = rl_min
                    best_bw = bw_frac
                    best_s11_db = s11_db
                    best_s11 = s11
                    best_zin = z_m
                    peak_idx = int(np.argmax(rl_band))
                    band_freq = freq_hz[band_mask]
                    best_f_peak = float(band_freq[peak_idx])
                    best_rl_peak = float(rl_band[peak_idx])
                    best = {
                        "topology": "pi",
                        "shunt_in_type": cin["type"],
                        "shunt_in_value": cin["value"],
                        "shunt_in_react_ohm": cin["x0"],
                        "series_type": cser["type"],
                        "series_value": cser["value"],
                        "series_react_ohm": cser["x0"],
                        "shunt_out_type": cout["type"],
                        "shunt_out_value": cout["value"],
                        "shunt_out_react_ohm": cout["x0"],
                        "rl_min_in_band_db": best_rl,
                        "rl_peak_in_band_db": best_rl_peak,
                        "f_peak_in_band_hz": best_f_peak,
                        "match_bandwidth_frac": best_bw,
                        "match_bandwidth_hz": bw_hz,
                        "f0_hz": f0_hz,
                    }

    if best is None or best_s11_db is None or best_s11 is None or best_zin is None:
        return None

    best["s11_db"] = best_s11_db
    best["s11"] = best_s11
    best["zin"] = best_zin
    return best


def matched_bandwidth(
    freq_hz: np.ndarray,
    rl_db: np.ndarray,
    f_low: float,
    f_high: float,
    rl_target: float = 10.0,
) -> Tuple[float, float]:
    if freq_hz.size == 0:
        return 0.0, 0.0
    mask = (freq_hz >= f_low) & (freq_hz <= f_high) & np.isfinite(rl_db)
    if not np.any(mask):
        return 0.0, 0.0
    band_freq = freq_hz[mask]
    band_rl = rl_db[mask]
    ok = band_rl >= rl_target
    if not np.any(ok):
        return 0.0, 0.0
    bw = float(np.max(band_freq[ok]) - np.min(band_freq[ok]))
    denom = max(f_high - f_low, 1e-9)
    return bw, bw / denom

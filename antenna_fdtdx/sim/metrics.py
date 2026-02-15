"""Metric extraction for FDTDX antenna simulations."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _complex_poynting(E: np.ndarray, H: np.ndarray) -> np.ndarray:
    # E, H shape: (3, Nx, Ny, Nz)
    return 0.5 * np.cross(E, np.conj(H), axis=0)


def complex_power_through_plane(
    E: np.ndarray,
    H: np.ndarray,
    axis: int,
    resolution_m: float,
) -> complex:
    poynting = _complex_poynting(E, H)
    comp = poynting[axis]
    area = resolution_m * resolution_m
    return np.sum(comp) * area


def positive_real_power_through_plane(
    E: np.ndarray,
    H: np.ndarray,
    axis: int,
    resolution_m: float,
    sign: float = 1.0,
) -> float:
    poynting = 0.5 * np.real(np.cross(E, np.conj(H), axis=0))
    comp = sign * poynting[axis]
    area = resolution_m * resolution_m
    return float(np.sum(np.maximum(comp, 0.0)) * area)


def real_power_through_plane(
    E: np.ndarray,
    H: np.ndarray,
    axis: int,
    resolution_m: float,
    sign: float = 1.0,
) -> float:
    poynting = 0.5 * np.real(np.cross(E, np.conj(H), axis=0))
    comp = sign * poynting[axis]
    area = resolution_m * resolution_m
    return float(np.sum(comp) * area)


def compute_port_impedance(
    Ez: np.ndarray,
    gap_height_m: float,
    port_power: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Ez shape: (num_freq,)
    V = Ez * gap_height_m
    V_abs2 = np.abs(V) ** 2
    zin = np.full_like(port_power, np.nan, dtype=np.complex128)
    valid = np.abs(port_power) > 1e-18
    zin[valid] = V_abs2[valid] / (2.0 * port_power[valid])
    return V, zin


def compute_gap_voltage(
    e_field: np.ndarray,
    gap_length_m: float,
    mask: np.ndarray | None = None,
    method: str = "mean",
) -> np.ndarray:
    if e_field.ndim != 4:
        raise ValueError("e_field must be (num_freq, nx, ny, nz)")
    if method not in {"mean", "centerline"}:
        raise ValueError(f"Unknown method: {method}")
    if method == "centerline":
        x_mid = e_field.shape[1] // 2
        y_mid = e_field.shape[2] // 2
        line = e_field[:, x_mid, y_mid, :]
        if mask is not None:
            if mask.shape != e_field.shape[1:]:
                raise ValueError("mask shape must match detector volume")
            mask_line = mask[x_mid, y_mid, :].astype(np.float32)
            mask_sum = np.sum(mask_line)
            if mask_sum <= 0:
                return np.full(e_field.shape[0], np.nan, dtype=np.complex128)
            e_mean = np.sum(line * mask_line[None, :], axis=1) / mask_sum
        else:
            e_mean = np.mean(line, axis=1)
        return -e_mean * gap_length_m
    if mask is not None:
        if mask.shape != e_field.shape[1:]:
            raise ValueError("mask shape must match detector volume")
        mask = mask.astype(np.float32)
        mask_sum = np.sum(mask)
        if mask_sum <= 0:
            return np.full(e_field.shape[0], np.nan, dtype=np.complex128)
        e_mean = np.sum(e_field * mask[None, ...], axis=(1, 2, 3)) / mask_sum
    else:
        e_mean = np.mean(e_field, axis=(1, 2, 3))
    return -e_mean * gap_length_m


def compute_loop_current(
    h0_field: np.ndarray,
    h1_field: np.ndarray,
    resolution_m: float,
    mask: np.ndarray | None = None,
    current_axis: int = 0,
) -> np.ndarray:
    if h0_field.ndim != 4 or h1_field.ndim != 4:
        raise ValueError("h0_field/h1_field must be (num_freq, nx, ny, nz)")
    if current_axis not in (0, 1, 2):
        raise ValueError("current_axis must be 0, 1, or 2")
    nx = h0_field.shape[1]
    ny = h0_field.shape[2]
    nz = h0_field.shape[3]
    if nx < 2 or ny < 2 or nz < 2:
        return np.full(h0_field.shape[0], np.nan, dtype=np.complex128)

    if current_axis == 0:
        idx = nx // 2
        h0_plane = h0_field[:, idx, :, :]
        h1_plane = h1_field[:, idx, :, :]
        mask_plane = None if mask is None else mask[idx, :, :]
    elif current_axis == 1:
        idx = ny // 2
        h0_plane = h0_field[:, :, idx, :]
        h1_plane = h1_field[:, :, idx, :]
        mask_plane = None if mask is None else mask[:, idx, :]
    else:
        idx = nz // 2
        h0_plane = h0_field[:, :, :, idx]
        h1_plane = h1_field[:, :, :, idx]
        mask_plane = None if mask is None else mask[:, :, idx]

    if mask_plane is not None:
        top = np.sum(h0_plane[:, :, -1] * mask_plane[:, -1][None, :], axis=1)
        bottom = np.sum(h0_plane[:, :, 0] * mask_plane[:, 0][None, :], axis=1)
        right = np.sum(h1_plane[:, -1, :] * mask_plane[-1, :][None, :], axis=1)
        left = np.sum(h1_plane[:, 0, :] * mask_plane[0, :][None, :], axis=1)
    else:
        top = np.sum(h0_plane[:, :, -1], axis=1)
        bottom = np.sum(h0_plane[:, :, 0], axis=1)
        right = np.sum(h1_plane[:, -1, :], axis=1)
        left = np.sum(h1_plane[:, 0, :], axis=1)
    return (top - bottom + left - right) * resolution_m


def compute_lumped_port_vi(
    phasor: np.ndarray,
    gap_length_m: float,
    resolution_m: float,
    mask: np.ndarray | None = None,
    current_axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    # phasor shape: (num_freq, num_comp, nx, ny, nz)
    if phasor.ndim != 5 or phasor.shape[1] < 3:
        raise ValueError("phasor must be (num_freq, num_comp>=3, nx, ny, nz)")
    if current_axis not in (0, 1, 2):
        raise ValueError("current_axis must be 0, 1, or 2")
    e_field = phasor[:, 0]
    h0_field = phasor[:, 1]
    h1_field = phasor[:, 2]
    V = compute_gap_voltage(e_field, gap_length_m, mask=mask)
    I = compute_loop_current(h0_field, h1_field, resolution_m, mask=mask, current_axis=current_axis)
    return V, I


def s11_from_impedance(zin: np.ndarray, z0: float = 50.0) -> np.ndarray:
    s11 = (zin - z0) / (zin + z0)
    return s11


def s11_db(s11: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(s11) + 1e-12)


def fb_ratio_db(p_fwd: float, p_back: float) -> float:
    if p_fwd <= 0 or p_back <= 0:
        return -200.0
    return 10.0 * math.log10(p_fwd / p_back)


def gain_db(p_fwd: float) -> float:
    if p_fwd <= 0:
        return -200.0
    return 10.0 * math.log10(p_fwd)


def summarize_band(freqs: np.ndarray, rl_db: np.ndarray) -> Dict[str, float]:
    idx = int(np.argmax(rl_db)) if rl_db.size else 0
    f_peak = float(freqs[idx]) if rl_db.size else float("nan")
    rl_peak = float(rl_db[idx]) if rl_db.size else float("nan")
    rl_min = float(np.min(rl_db)) if rl_db.size else float("nan")
    return {
        "f_peak_hz": f_peak,
        "rl_peak_db": rl_peak,
        "rl_min_in_band_db": rl_min,
    }


def summarize_sweep(
    freqs: np.ndarray,
    rl_db: np.ndarray,
    f_low: float,
    f_high: float,
) -> Dict[str, float]:
    if freqs.size == 0:
        return {
            "f_peak_hz": float("nan"),
            "rl_peak_db": float("nan"),
            "rl_min_in_band_db": float("nan"),
        }
    idx = int(np.argmax(rl_db))
    f_peak = float(freqs[idx])
    rl_peak = float(rl_db[idx])
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        rl_min = float("nan")
    else:
        rl_min = float(np.min(rl_db[mask]))
    return {
        "f_peak_hz": f_peak,
        "rl_peak_db": rl_peak,
        "rl_min_in_band_db": rl_min,
    }

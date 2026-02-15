"""Port calibration helpers (one-port SOL)."""

from __future__ import annotations

import json
import math
from typing import Dict, Tuple

import numpy as np


def _as_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)


def solve_oneport_calibration(
    s_true: np.ndarray,
    s_meas: np.ndarray,
    min_det: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a one-port error model for each frequency.

    Uses a bilinear transform: s_meas = (A + B*s_true) / (1 + C*s_true)
    Returns A, B, C arrays plus a boolean valid mask.
    """
    s_true = np.asarray(s_true, dtype=np.complex128)
    s_meas = np.asarray(s_meas, dtype=np.complex128)
    if s_true.shape != s_meas.shape:
        raise ValueError("s_true and s_meas must have the same shape")
    if s_true.ndim != 2 or s_true.shape[0] < 3:
        raise ValueError("s_true/s_meas must be (nstd>=3, nfreq)")

    nstd, nfreq = s_true.shape
    A = np.full(nfreq, np.nan + 1j * np.nan, dtype=np.complex128)
    B = np.full(nfreq, np.nan + 1j * np.nan, dtype=np.complex128)
    C = np.full(nfreq, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.zeros(nfreq, dtype=bool)

    for idx in range(nfreq):
        st = s_true[:, idx]
        sm = s_meas[:, idx]
        if not np.all(np.isfinite(st)) or not np.all(np.isfinite(sm)):
            continue
        mat = np.column_stack([np.ones(nstd, dtype=np.complex128), st, -sm * st])
        if nstd == 3:
            det = np.linalg.det(mat)
            if not np.isfinite(det) or abs(det) < min_det:
                continue
            try:
                sol = np.linalg.solve(mat, sm)
            except np.linalg.LinAlgError:
                continue
            rank = 3
        else:
            sol, _, rank, _ = np.linalg.lstsq(mat, sm, rcond=None)
        if rank < 3:
            continue
        if not np.all(np.isfinite(sol)):
            continue
        A[idx], B[idx], C[idx] = sol
        valid[idx] = True
    return A, B, C, valid


def apply_oneport_calibration(
    s_meas: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Apply the one-port calibration to measured s-parameters."""
    s_meas = np.asarray(s_meas, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    if s_meas.ndim == 1:
        A_b = A
        B_b = B
        C_b = C
    elif s_meas.ndim == 2:
        A_b = A[None, :]
        B_b = B[None, :]
        C_b = C[None, :]
    else:
        raise ValueError("s_meas must be 1D or 2D")
    denom = B_b - C_b * s_meas
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-18) & np.isfinite(A_b)
    numer = s_meas - A_b
    s_cal = np.where(valid, numer / denom, np.nan + 1j * np.nan)
    return s_cal.astype(np.complex128)


def load_calibration(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    freq = np.asarray(payload["freq_hz"], dtype=np.float64)
    A = _as_complex(payload["a_real"], payload["a_imag"])
    B = _as_complex(payload["b_real"], payload["b_imag"])
    C = _as_complex(payload["c_real"], payload["c_imag"])
    valid = np.asarray(payload.get("valid_mask", []), dtype=bool)
    if valid.size == 0:
        valid = np.ones_like(freq, dtype=bool)
    return {
        "freq_hz": freq,
        "A": A,
        "B": B,
        "C": C,
        "valid": valid,
        "metric": payload.get("metric"),
        "fixture": payload.get("fixture"),
    }


def save_calibration(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def interp_calibration(cal: Dict[str, np.ndarray], freq_hz: np.ndarray) -> Dict[str, np.ndarray]:
    """Interpolate calibration coefficients onto a new frequency grid."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    src_freq = np.asarray(cal["freq_hz"], dtype=np.float64)
    if freq_hz.shape == src_freq.shape and np.allclose(freq_hz, src_freq, rtol=0, atol=0):
        return cal

    def _interp_complex(arr: np.ndarray) -> np.ndarray:
        real = np.interp(freq_hz, src_freq, np.real(arr))
        imag = np.interp(freq_hz, src_freq, np.imag(arr))
        return real + 1j * imag

    out = dict(cal)
    out["freq_hz"] = freq_hz
    out["A"] = _interp_complex(cal["A"])
    out["B"] = _interp_complex(cal["B"])
    out["C"] = _interp_complex(cal["C"])
    out["valid"] = np.asarray(cal.get("valid", np.ones_like(src_freq, dtype=bool)))
    if out["valid"].shape != freq_hz.shape:
        valid = np.interp(freq_hz, src_freq, out["valid"].astype(float))
        out["valid"] = valid >= 0.5
    return out


def load_port_reference(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    freq = np.asarray(payload["freq_hz"], dtype=np.float64)
    v_open = _as_complex(payload["v_open_real"], payload["v_open_imag"])
    i_short = _as_complex(payload["i_short_real"], payload["i_short_imag"])
    return {
        "freq_hz": freq,
        "v_open": v_open,
        "i_short": i_short,
        "model": payload.get("model"),
        "quality": payload.get("quality"),
        "method": payload.get("method", "thevenin"),
    }


def save_port_reference(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def interp_port_reference(ref: Dict[str, np.ndarray], freq_hz: np.ndarray) -> Dict[str, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    src_freq = np.asarray(ref["freq_hz"], dtype=np.float64)
    if freq_hz.shape == src_freq.shape and np.allclose(freq_hz, src_freq, rtol=0, atol=0):
        return ref

    def _interp_complex(arr: np.ndarray) -> np.ndarray:
        real = np.interp(freq_hz, src_freq, np.real(arr))
        imag = np.interp(freq_hz, src_freq, np.imag(arr))
        return real + 1j * imag

    out = dict(ref)
    out["freq_hz"] = freq_hz
    out["v_open"] = _interp_complex(ref["v_open"])
    out["i_short"] = _interp_complex(ref["i_short"])
    return out


def thevenin_s11(
    v_load: np.ndarray,
    v_open: np.ndarray,
    i_short: np.ndarray,
    z0: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute load impedance and S11 via a Thevenin open/short reference."""
    v_load = np.asarray(v_load, dtype=np.complex128)
    v_open = np.asarray(v_open, dtype=np.complex128)
    i_short = np.asarray(i_short, dtype=np.complex128)
    z_th = np.full_like(v_open, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_th = np.isfinite(v_open) & np.isfinite(i_short) & (np.abs(i_short) > 1e-18)
    z_th[valid_th] = v_open[valid_th] / i_short[valid_th]
    denom = v_open - v_load
    z_load = np.full_like(v_open, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_load = np.isfinite(z_th) & np.isfinite(v_load) & np.isfinite(denom) & (np.abs(denom) > 1e-18)
    z_load[valid_load] = z_th[valid_load] * v_load[valid_load] / denom[valid_load]
    near_open = np.isfinite(z_th) & np.isfinite(v_load) & np.isfinite(denom) & (np.abs(denom) <= 1e-18)
    if np.any(near_open):
        z_load[near_open] = 1e9 + 0j
    s11 = (z_load - z0) / (z_load + z0)
    return z_load, s11

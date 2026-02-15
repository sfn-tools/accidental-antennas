"""Post-processing helpers for openEMS results."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _get_wave_attr(port, field: str, name: str):
    if hasattr(port, field):
        obj = getattr(port, field)
        if hasattr(obj, name):
            return getattr(obj, name)
    legacy = f"{field}_{name}"
    if hasattr(port, legacy):
        return getattr(port, legacy)
    raise AttributeError(f"Port missing {field}.{name} or {legacy}")


def _port_uf(port, name: str):
    return _get_wave_attr(port, "uf", name)


def _port_if(port, name: str):
    return _get_wave_attr(port, "if", name)


def calc_sparams(ports, sim_path: str, freq: np.ndarray, ref_impedance: float, excite_port: int = 0):
    for port in ports:
        port.CalcPort(sim_path, freq, ref_impedance=ref_impedance)
    base = _port_uf(ports[excite_port], "inc")
    base = np.where(np.abs(base) < 1e-30, 1e-30 + 0j, base)
    sparams = []
    for port in ports:
        sparams.append(_port_uf(port, "ref") / base)
    return sparams


def port_s11_zin(port) -> Tuple[np.ndarray, np.ndarray]:
    uf_inc = _port_uf(port, "inc")
    uf_ref = _port_uf(port, "ref")
    uf_tot = _port_uf(port, "tot")
    if_tot = _port_if(port, "tot")

    uf_inc = np.where(np.abs(uf_inc) < 1e-30, 1e-30 + 0j, uf_inc)
    if_tot = np.where(np.abs(if_tot) < 1e-30, 1e-30 + 0j, if_tot)
    s11 = uf_ref / uf_inc
    zin = uf_tot / if_tot
    return s11, zin


def s11_db(s11: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(s11) + 1e-12)


def port_power_acc(port) -> np.ndarray:
    if hasattr(port, "P_acc"):
        return np.real(port.P_acc)
    uf_tot = _port_uf(port, "tot")
    if_tot = _port_if(port, "tot")
    return 0.5 * np.real(uf_tot * np.conj(if_tot))


def port_power_inc(port) -> np.ndarray:
    if hasattr(port, "P_inc"):
        return np.real(port.P_inc)
    uf_inc = _port_uf(port, "inc")
    if_inc = _port_if(port, "inc")
    return 0.5 * np.real(uf_inc * np.conj(if_inc))


def port_power_ref(port) -> np.ndarray:
    if hasattr(port, "P_ref"):
        return np.real(port.P_ref)
    uf_ref = _port_uf(port, "ref")
    if_ref = _port_if(port, "ref")
    return 0.5 * np.real(uf_ref * np.conj(if_ref))


def _interp_at(freq: np.ndarray, values: np.ndarray, f0: float) -> float:
    if len(freq) == 0:
        return 0.0
    return float(np.interp(f0, freq, values))


def port_power_acc_at(port, freq: np.ndarray, f0: float) -> float:
    p_acc = port_power_acc(port)
    return _interp_at(freq, p_acc, f0)


def port_power_inc_at(port, freq: np.ndarray, f0: float) -> float:
    p_inc = port_power_inc(port)
    return _interp_at(freq, p_inc, f0)


def port_power_ref_at(port, freq: np.ndarray, f0: float) -> float:
    p_ref = port_power_ref(port)
    return _interp_at(freq, p_ref, f0)


def calc_realized_gain_db(nf2ff_res, p_acc: float, theta_idx: int, phi_idx: int) -> float:
    if p_acc <= 0:
        return -200.0
    u = float(nf2ff_res.P_rad[0][theta_idx, phi_idx])
    realized = 4.0 * math.pi * u / p_acc
    return 10.0 * math.log10(max(realized, 1e-12))


def save_s11_csv(path: str, freq: np.ndarray, s11_db_vals: np.ndarray, zin: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["freq_hz", "s11_db", "return_loss_db", "zin_real_ohm", "zin_imag_ohm"]
        )
        for f, s, z in zip(freq, s11_db_vals, zin):
            writer.writerow([f, s, -s, float(np.real(z)), float(np.imag(z))])


def save_s11_plot(path: str, freq: np.ndarray, s11_db_vals: np.ndarray) -> None:
    try:
        import os
        mpl_cfg = os.environ.get("MPLCONFIGDIR")
        if not mpl_cfg:
            mpl_cfg = os.path.join(os.getcwd(), ".mplconfig")
            os.makedirs(mpl_cfg, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = mpl_cfg
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(freq / 1e9, s11_db_vals, "k-")
    plt.grid(True)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Reflection coefficient")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_smith_csv(path: str, freq: np.ndarray, s11: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_hz", "gamma_real", "gamma_imag", "s11_db", "return_loss_db"])
        s11_db_vals = 20.0 * np.log10(np.abs(s11) + 1e-12)
        for f, g, s_db in zip(freq, s11, s11_db_vals):
            writer.writerow([f, float(np.real(g)), float(np.imag(g)), float(s_db), float(-s_db)])


def calc_directivity_db(nf2ff_res, theta_idx: int, phi_idx: int) -> float:
    prad = float(nf2ff_res.Prad[0])
    if prad <= 0:
        return -200.0
    u = float(nf2ff_res.P_rad[0][theta_idx, phi_idx])
    directivity = 4.0 * math.pi * u / prad
    return 10.0 * math.log10(max(directivity, 1e-12))


def calc_plane_pattern(
    nf2ff,
    sim_path: str,
    f0_hz: float,
    phi_deg: Iterable[float],
    outfile: str = "nf2ff_plane.h5",
) -> Tuple[np.ndarray, np.ndarray]:
    theta = [90.0]
    phi_list = list(phi_deg)
    res = nf2ff.CalcNF2FF(
        sim_path,
        f0_hz,
        theta,
        phi_list,
        radius=1,
        center=[0, 0, 0],
        outfile=outfile,
        read_cached=False,
    )
    gains = []
    for idx_phi in range(len(phi_list)):
        gains.append(calc_directivity_db(res, 0, idx_phi))
    return np.array(phi_list), np.array(gains)


def save_pattern_csv(path: str, phi_deg: np.ndarray, gain_db: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi_deg", "gain_dbi"])
        for p, g in zip(phi_deg, gain_db):
            writer.writerow([p, g])


def save_metrics(path: str, metrics: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def save_summary(path: str, metrics: Dict) -> None:
    lines = []
    for key in sorted(metrics.keys()):
        lines.append(f"{key}: {metrics[key]}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

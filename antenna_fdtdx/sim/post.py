"""File output helpers for FDTDX antenna runs."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict

import numpy as np


def save_s11_csv(path: str, freq: np.ndarray, s11_db_vals: np.ndarray, zin: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_hz", "s11_db", "return_loss_db", "zin_real_ohm", "zin_imag_ohm"])
        for f, s, z in zip(freq, s11_db_vals, zin):
            writer.writerow([f, s, -s, float(np.real(z)), float(np.imag(z))])


def save_s11_plot(path: str, freq: np.ndarray, s11_db_vals: np.ndarray) -> None:
    try:
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


def save_metrics(path: str, metrics: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def save_summary(path: str, metrics: Dict) -> None:
    lines = []
    for key in sorted(metrics.keys()):
        lines.append(f"{key}: {metrics[key]}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

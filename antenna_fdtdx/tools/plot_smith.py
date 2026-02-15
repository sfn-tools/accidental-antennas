#!/usr/bin/env python3
"""Plot a Smith chart from s11_matched_smith.csv."""

from __future__ import annotations

import argparse
import csv
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> Tuple[List[float], List[float], List[float]]:
    gamma_r: List[float] = []
    gamma_i: List[float] = []
    freqs: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gamma_r.append(float(row["gamma_real"]))
            gamma_i.append(float(row["gamma_imag"]))
            freq = row.get("freq_hz")
            freqs.append(float(freq) if freq not in (None, "") else float("nan"))
    return gamma_r, gamma_i, freqs


def _circle(cx: float, cy: float, radius: float, n: int = 361) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    return cx + radius * np.cos(theta), cy + radius * np.sin(theta)


def _draw_grid(ax) -> None:
    grid_color = "#cccccc"
    unit_x, unit_y = _circle(0.0, 0.0, 1.0)
    ax.plot(unit_x, unit_y, color=grid_color, lw=1.0)
    ax.axhline(0.0, color=grid_color, lw=0.5)
    ax.axvline(0.0, color=grid_color, lw=0.5)

    r_vals = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for r in r_vals:
        cx = r / (r + 1.0)
        radius = 1.0 / (r + 1.0)
        x, y = _circle(cx, 0.0, radius)
        ax.plot(x, y, color=grid_color, lw=0.5)

    x_vals = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for x_val in x_vals:
        for sign in (1.0, -1.0):
            x = sign * x_val
            cx = 1.0
            cy = 1.0 / x
            radius = 1.0 / abs(x)
            gx, gy = _circle(cx, cy, radius)
            mask = gx * gx + gy * gy <= 1.0 + 1e-9
            ax.plot(gx[mask], gy[mask], color=grid_color, lw=0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to s11_matched_smith.csv")
    parser.add_argument("--output", default="smith.png", help="Output image path")
    parser.add_argument("--title", default="Smith chart", help="Plot title")
    parser.add_argument("--color-freq", action="store_true", help="Color by frequency")
    args = parser.parse_args()

    gamma_r, gamma_i, freqs = _read_csv(args.input)

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_grid(ax)

    if args.color_freq and any(math.isfinite(f) for f in freqs):
        ax.scatter(gamma_r, gamma_i, c=freqs, cmap="viridis", s=12)
    else:
        ax.plot(gamma_r, gamma_i, "-o", ms=2, lw=1.0, color="#1f77b4")

    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Gamma real")
    ax.set_ylabel("Gamma imag")
    ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

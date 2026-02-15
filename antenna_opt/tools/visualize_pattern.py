"""Generate a 3D far-field pattern VTK from a run folder."""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from sim import model_common as mc


def write_vtk(path: str, theta_deg, phi_deg, gain_db):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")
    gain_lin = 10 ** (gain_db / 10.0)
    r = gain_lin / np.max(gain_lin)

    x = r * np.sin(th_grid) * np.cos(ph_grid)
    y = r * np.sin(th_grid) * np.sin(ph_grid)
    z = r * np.cos(th_grid)

    n_theta, n_phi = r.shape
    points = []
    scalars = []
    for i in range(n_theta):
        for j in range(n_phi):
            points.append((x[i, j], y[i, j], z[i, j]))
            scalars.append(gain_db[i, j])

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 2.0\n")
        handle.write("openEMS NF2FF pattern\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_GRID\n")
        handle.write(f"DIMENSIONS {n_phi} {n_theta} 1\n")
        handle.write(f"POINTS {n_theta * n_phi} float\n")
        for pt in points:
            handle.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
        handle.write(f"\nPOINT_DATA {n_theta * n_phi}\n")
        handle.write("SCALARS gain_dbi float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for val in scalars:
            handle.write(f"{val:.6f}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--f0", type=float, required=True, help="Frequency in Hz")
    parser.add_argument("--theta-step", type=float, default=5.0)
    parser.add_argument("--phi-step", type=float, default=5.0)
    parser.add_argument("--out", required=True, help="Output VTK path")
    args = parser.parse_args()

    try:
        from openEMS import nf2ff as nf2ff_mod
        from CSXCAD import ContinuousStructure
    except Exception:
        print(
            "openEMS Python modules not available.\n"
            "Activate your openEMS environment first (for example):\n"
            "  source ../scripts/env.sh",
            file=sys.stderr,
        )
        return 1

    meta_path = os.path.join(args.run, "meta.json")
    if not os.path.exists(meta_path):
        print("meta.json not found in run directory.", file=sys.stderr)
        return 1

    meta = mc.load_json(meta_path)
    nf2ff_meta = meta.get("nf2ff", {})
    start = nf2ff_meta.get("nf2ff_start_mm")
    stop = nf2ff_meta.get("nf2ff_stop_mm")
    if not start or not stop:
        print("nf2ff metadata missing from meta.json.", file=sys.stderr)
        return 1

    theta = np.arange(0.0, 180.0 + args.theta_step, args.theta_step)
    phi = np.arange(0.0, 360.0 + args.phi_step, args.phi_step)

    CSX = ContinuousStructure()
    nf2ff_box = nf2ff_mod.nf2ff(CSX, "nf2ff", start, stop)
    res = nf2ff_box.CalcNF2FF(
        args.run,
        args.f0,
        theta,
        phi,
        radius=1,
        center=[0, 0, 0],
        outfile=os.path.basename(args.out) + ".h5",
        read_cached=False,
    )

    prad = float(res.Prad[0])
    if prad <= 0:
        print("Invalid radiated power in NF2FF result.", file=sys.stderr)
        return 1

    u = res.P_rad[0]
    directivity = 4.0 * math.pi * u / prad
    gain_db = 10.0 * np.log10(np.maximum(directivity, 1e-12))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_vtk(args.out, theta, phi, gain_db)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

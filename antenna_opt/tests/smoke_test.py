"""Smoke test for openEMS Python interface."""

from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    try:
        from openEMS import openEMS, ports
        from openEMS import physical_constants as pc
        from CSXCAD import ContinuousStructure
    except Exception:
        print(
            "openEMS Python modules not available.\n"
            "Activate your openEMS environment first (for example):\n"
            "  source ../scripts/env.sh",
            file=sys.stderr,
        )
        return 1

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_dir = os.path.join(root, "runs", "smoke_test")
    os.makedirs(run_dir, exist_ok=True)

    FDTD = openEMS(EndCriteria=1e-3, NrTS=1000)
    FDTD.SetGaussExcite(2.45e9, 1.0e9)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.SetLines("x", np.array([-10, -2, 0, 2, 10]))
    grid.SetLines("y", np.array([-10, -2, 0, 2, 10]))
    grid.SetLines("z", np.array([-5, 0, 0.8, 5]))

    sub = CSX.AddMaterial("FR4")
    kappa = 2 * np.pi * 2.45e9 * pc.EPS0 * 4.3 * 0.02
    sub.SetMaterialProperty(epsilon=4.3, kappa=kappa)
    sub.AddBox([-10, -10, 0], [10, 10, 0.8])

    gnd = CSX.AddMetal("gnd")
    gnd.AddBox([-10, -10, 0], [10, 10, 0])

    patch = CSX.AddMetal("patch")
    patch.AddBox([-2, -2, 0.8], [2, 2, 0.8])

    port = ports.LumpedPort(
        CSX,
        port_nr=1,
        R=50,
        start=[0, 0, 0],
        stop=[0, 0, 0.8],
        exc_dir="z",
        excite=1,
    )

    FDTD.SetCSX(CSX)
    FDTD.Run(run_dir, cleanup=True, verbose=0)

    freq = np.linspace(2.0e9, 3.0e9, 3)
    port.CalcPort(run_dir, freq, ref_impedance=50)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-12)
    print("Smoke test S11 (dB):", s11_db)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

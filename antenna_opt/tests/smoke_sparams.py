"""Smoke tests for S-parameter extraction."""

from __future__ import annotations

import os
import sys

import numpy as np

from sim import post


def _build_parallel_plate(CSX, length_mm: float, width_mm: float, sep_mm: float) -> None:
    metal = CSX.AddMetal("pec")
    metal.AddBox([0, -width_mm / 2.0, 0], [length_mm, width_mm / 2.0, 0])
    metal.AddBox([0, -width_mm / 2.0, sep_mm], [length_mm, width_mm / 2.0, sep_mm])


def _run_case(run_dir: str, port_resistance: float, self_match: bool) -> np.ndarray:
    from openEMS import openEMS, ports
    from CSXCAD import ContinuousStructure

    length_mm = 40.0
    width_mm = 10.0
    sep_mm = 1.0
    freq = np.linspace(1.0e9, 4.0e9, 5)

    FDTD = openEMS(EndCriteria=1e-3, NrTS=8000)
    FDTD.SetGaussExcite(2.5e9, 1.5e9)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.SetLines("x", np.linspace(0.0, length_mm, 41))
    grid.SetLines("y", np.linspace(-width_mm / 2.0, width_mm / 2.0, 21))
    grid.SetLines("z", np.linspace(0.0, sep_mm + 6.0, 17))

    _build_parallel_plate(CSX, length_mm, width_mm, sep_mm)

    port = ports.LumpedPort(
        CSX,
        port_nr=1,
        R=port_resistance,
        start=[5.0, 0.0, 0.0],
        stop=[5.0, 0.0, sep_mm],
        exc_dir="z",
        excite=1,
    )

    FDTD.SetCSX(CSX)
    FDTD.Run(run_dir, cleanup=True, verbose=0)

    port.CalcPort(run_dir, freq, ref_impedance=50)
    if self_match:
        zin = port.uf_tot / port.if_tot
        port.CalcPort(run_dir, freq, ref_impedance=zin)

    s11 = port.uf_ref / port.uf_inc
    return post.s11_db(s11)


def main() -> int:
    try:
        import openEMS  # noqa: F401
        import CSXCAD  # noqa: F401
    except Exception:
        print(
            "openEMS Python modules not available.\n"
            "Activate your openEMS environment first (for example):\n"
            "  source ../scripts/env.sh",
            file=sys.stderr,
        )
        return 1

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_root = os.path.join(root, "runs")
    os.makedirs(run_root, exist_ok=True)

    matched_dir = os.path.join(run_root, "smoke_sparams_matched")
    open_dir = os.path.join(run_root, "smoke_sparams_open")

    s11_matched = _run_case(matched_dir, port_resistance=50.0, self_match=True)
    if not np.all(s11_matched < -20.0):
        raise AssertionError(f"Matched case S11 too high: {s11_matched}")

    s11_open = _run_case(open_dir, port_resistance=1e9, self_match=False)
    if not np.all(s11_open > -3.0):
        raise AssertionError(f"Open case S11 not near 0 dB: {s11_open}")

    print("S-parameter smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

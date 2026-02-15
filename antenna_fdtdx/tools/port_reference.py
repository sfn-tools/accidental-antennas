"""Generate a Thevenin port reference (open/short) for FDTDX models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np

from sim import calibration
from sim.run_one import run_one


def _load_port_vi(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = []
    v_vals = []
    i_vals = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline()
        if "freq_hz" not in header:
            raise ValueError(f"Unexpected port_vi.csv header: {header.strip()}")
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            freq.append(float(parts[0]))
            v_vals.append(float(parts[1]) + 1j * float(parts[2]))
            i_vals.append(float(parts[3]) + 1j * float(parts[4]))
    if not freq:
        raise ValueError(f"No data in {path}")
    return np.asarray(freq), np.asarray(v_vals), np.asarray(i_vals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ModelConfig name")
    parser.add_argument("--quality", choices=["coarse", "mid", "fast", "medium", "fine", "high"], default="coarse")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--open-init", choices=["empty", "patch", "uniform", "solid"], default="empty")
    parser.add_argument("--short-init", choices=["solid", "patch", "uniform", "empty"], default="solid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default=None, help="Root folder for reference runs")
    parser.add_argument("--out", default=None, help="Output reference JSON path")
    parser.add_argument("--lossy", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--port-vi-mode", choices=["loop", "power"], default="loop")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_root = args.run_root or os.path.join(repo_root, "runs_port_ref")

    open_metrics = run_one(
        model_name=args.model,
        quality=args.quality,
        backend=args.backend,
        init_mode=args.open_init,
        seed=args.seed,
        run_root=run_root,
        force=args.force,
        use_loss=args.lossy,
        time_scale=args.time_scale,
        rl_mode="vi",
        port_vi_mode=args.port_vi_mode,
        save_port_vi=True,
    )
    short_metrics = run_one(
        model_name=args.model,
        quality=args.quality,
        backend=args.backend,
        init_mode=args.short_init,
        seed=args.seed + 1,
        run_root=run_root,
        force=args.force,
        use_loss=args.lossy,
        time_scale=args.time_scale,
        rl_mode="vi",
        port_vi_mode=args.port_vi_mode,
        save_port_vi=True,
    )

    open_vi = os.path.join(open_metrics["run_dir"], "port_vi.csv")
    short_vi = os.path.join(short_metrics["run_dir"], "port_vi.csv")
    if not os.path.exists(open_vi):
        raise FileNotFoundError(f"Missing port_vi.csv for open reference: {open_vi}")
    if not os.path.exists(short_vi):
        raise FileNotFoundError(f"Missing port_vi.csv for short reference: {short_vi}")

    f_open, v_open, _ = _load_port_vi(open_vi)
    f_short, _, i_short = _load_port_vi(short_vi)
    if f_open.shape != f_short.shape or not np.allclose(f_open, f_short):
        raise ValueError("Open/short frequency grids do not match")

    payload = {
        "model": args.model,
        "quality": args.quality,
        "backend": args.backend,
        "method": "thevenin",
        "open_init": args.open_init,
        "short_init": args.short_init,
        "freq_hz": [float(f) for f in f_open],
        "v_open_real": [float(np.real(v)) for v in v_open],
        "v_open_imag": [float(np.imag(v)) for v in v_open],
        "i_short_real": [float(np.real(i)) for i in i_short],
        "i_short_imag": [float(np.imag(i)) for i in i_short],
        "open_run_id": open_metrics.get("run_id"),
        "short_run_id": short_metrics.get("run_id"),
    }

    out_path = args.out or os.path.join(repo_root, "calibration", f"port_reference_{args.model}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    calibration.save_port_reference(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

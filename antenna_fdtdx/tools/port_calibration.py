"""Generate a one-port SOL calibration for FDTDX port extraction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim import calibration
from sim import models


def _load_openems_s11(path: str, source: str, z0: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    freq: list[float] = []
    s11: list[complex] = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline()
        if "freq_hz" not in header:
            raise ValueError(f"Unexpected S11 header: {header.strip()}")
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            freq.append(float(parts[0]))
            if source == "zin":
                zr = float(parts[3])
                zi = float(parts[4])
                z = zr + 1j * zi
                s11.append((z - z0) / (z + z0))
            else:
                gr = float(parts[1])
                gi = float(parts[2])
                s11.append(gr + 1j * gi)
    if not freq:
        raise ValueError(f"No data found in {path}")
    freq_arr = np.asarray(freq, dtype=np.float64)
    s11_arr = np.asarray(s11, dtype=np.complex128)
    return freq_arr, s11_arr


def _interp_s11(freq_src: np.ndarray, s11_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    if freq_src.shape == freq_dst.shape and np.allclose(freq_src, freq_dst):
        return s11_src
    real = np.interp(freq_dst, freq_src, np.real(s11_src))
    imag = np.interp(freq_dst, freq_src, np.imag(s11_src))
    return real + 1j * imag


def _mask_s11_mag(s11: np.ndarray, max_mag: float | None) -> np.ndarray:
    if max_mag is None or max_mag <= 0:
        return np.asarray(s11, dtype=np.complex128)
    s11 = np.asarray(s11, dtype=np.complex128)
    bad = np.abs(s11) > float(max_mag)
    if not np.any(bad):
        return s11
    masked = s11.copy()
    masked[bad] = np.nan + 1j * np.nan
    return masked


def _run_openems(
    python_bin: str,
    antenna_opt_root: str,
    out_root: str,
    fmin_hz: float,
    fmax_hz: float,
    f0_hz: float,
    points: int,
    load_sigma: float | None = None,
    load_modes: str = "open,short,match",
    port_types: str = "lumped",
    extra_args: List[str] | None = None,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "tools.baseline_suite",
        "--case",
        "microstrip",
        "--port-types",
        port_types,
        "--load-modes",
        load_modes,
        "--fmin-hz",
        str(float(fmin_hz)),
        "--fmax-hz",
        str(float(fmax_hz)),
        "--f0-hz",
        str(float(f0_hz)),
        "--points",
        str(int(points)),
        "--out-root",
        out_root,
        "--summary",
        os.path.join(out_root, "summary_microstrip.json"),
        "--prune",
    ]
    if load_sigma is not None:
        cmd.extend(["--load-sigma", str(float(load_sigma))])
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=antenna_opt_root, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "openEMS baseline failed"
        raise RuntimeError(msg)


def _s11_from_case(case: Dict, metric: str) -> np.ndarray:
    key_real = f"s11_{metric}_real"
    key_imag = f"s11_{metric}_imag"
    if key_real not in case or key_imag not in case:
        raise KeyError(f"Missing {key_real}/{key_imag}")
    real = np.asarray(case[key_real], dtype=np.float64)
    imag = np.asarray(case[key_imag], dtype=np.float64)
    return real + 1j * imag


def _complex_from_case(case: Dict, prefix: str) -> np.ndarray:
    key_real = f"{prefix}_real"
    key_imag = f"{prefix}_imag"
    if key_real not in case or key_imag not in case:
        raise KeyError(f"Missing {key_real}/{key_imag}")
    real = np.asarray(case[key_real], dtype=np.float64)
    imag = np.asarray(case[key_imag], dtype=np.float64)
    return real + 1j * imag


def _parse_sigmas(text: str | None) -> List[float]:
    if text is None:
        return []
    values = [part.strip() for part in text.split(",") if part.strip()]
    sigmas: List[float] = []
    for val in values:
        try:
            sigmas.append(float(val))
        except ValueError as exc:
            raise ValueError(f"Invalid sigma value: {val}") from exc
    return sigmas


def _sigma_tag(sigma: float) -> str:
    return f"{sigma:.4g}".replace(".", "p").replace("-", "m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Optional ModelConfig name to set sweep range")
    parser.add_argument("--fmin-hz", type=float, default=2.0e9)
    parser.add_argument("--fmax-hz", type=float, default=8.0e9)
    parser.add_argument("--f0-hz", type=float, default=None)
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument("--metric", choices=["vi", "flux", "thevenin"], default="vi")
    parser.add_argument(
        "--cal-sigmas",
        default=None,
        help="Comma-separated list of match load sigma values for calibration",
    )
    parser.add_argument("--validate-sigma", type=float, default=None, help="Optional sigma for validation load")
    parser.add_argument("--openems-python", default="python3")
    parser.add_argument("--openems-root", default=None)
    parser.add_argument("--openems-args", default=None, help="Extra args passed to antenna_opt.tools.baseline_suite")
    parser.add_argument("--openems-port-types", default="lumped", help="openEMS microstrip port types (lumped or msl)")
    parser.add_argument(
        "--openems-s11-source",
        choices=["zin", "wave"],
        default="zin",
        help="Use openEMS s11.csv (zin) or s11_wave.csv (wave) as calibration reference.",
    )
    parser.add_argument(
        "--s11-max-mag",
        type=float,
        default=1.05,
        help="Mask |S11| above this magnitude for standards (<=0 disables).",
    )
    parser.add_argument("--out", default=None, help="Output calibration JSON path")
    parser.add_argument("--skip-openems", action="store_true")
    parser.add_argument("--fdtdx-source-offset-mm", type=float, default=0.0)
    parser.add_argument("--fdtdx-port-offset-mm", type=float, default=0.0)
    parser.add_argument("--fdtdx-port-len-mm", type=float, default=2.0)
    parser.add_argument("--fdtdx-resolution-mm", type=float, default=0.4)
    parser.add_argument("--fdtdx-port-vi-mode", choices=["loop", "power"], default="loop")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    antenna_opt_root = args.openems_root or os.path.join(repo_root, "antenna_opt")
    out_root = os.path.join(repo_root, "antenna_opt", "reports", "baseline_suite_cal")

    if args.model:
        model = models.MODEL_CONFIGS[args.model]
        freqs = model.sweep_freqs(points=args.points)
        fmin_hz = min(freqs)
        fmax_hz = max(freqs)
        f0_hz = model.f0_hz
    else:
        fmin_hz = float(args.fmin_hz)
        fmax_hz = float(args.fmax_hz)
        f0_hz = float(args.f0_hz) if args.f0_hz is not None else 0.5 * (fmin_hz + fmax_hz)
        freqs = np.linspace(fmin_hz, fmax_hz, int(args.points)).tolist()

    cal_sigmas = _parse_sigmas(args.cal_sigmas)
    if not cal_sigmas:
        cal_sigmas = [3.3]
    cal_sigmas = sorted({float(s) for s in cal_sigmas})

    base_root = os.path.join(out_root, "base")
    extra_args = args.openems_args.split() if args.openems_args else None
    if not args.skip_openems:
        _run_openems(
            args.openems_python,
            antenna_opt_root,
            base_root,
            fmin_hz,
            fmax_hz,
            f0_hz,
            len(freqs),
            load_sigma=cal_sigmas[0],
            load_modes="open,short",
            port_types=args.openems_port_types,
            extra_args=extra_args,
        )
        for sigma in cal_sigmas:
            sigma_root = os.path.join(out_root, f"sigma_{_sigma_tag(sigma)}")
            _run_openems(
                args.openems_python,
                antenna_opt_root,
                sigma_root,
                fmin_hz,
                fmax_hz,
                f0_hz,
                len(freqs),
                load_sigma=float(sigma),
                load_modes="match",
                port_types=args.openems_port_types,
                extra_args=extra_args,
            )

    openems = {}
    port_tag = args.openems_port_types.split(",")[0].strip()
    for mode in ("open", "short"):
        s11_name = "s11.csv" if args.openems_s11_source == "zin" else "s11_wave.csv"
        s11_path = os.path.join(base_root, f"microstrip_{port_tag}_{mode}", s11_name)
        if not os.path.exists(s11_path):
            raise FileNotFoundError(f"Missing openEMS s11.csv for {mode}: {s11_path}")
        freq_src, s11_src = _load_openems_s11(s11_path, args.openems_s11_source)
        openems[mode] = _mask_s11_mag(
            _interp_s11(freq_src, s11_src, np.asarray(freqs, dtype=np.float64)),
            args.s11_max_mag,
        )

    openems_matches = []
    for sigma in cal_sigmas:
        sigma_root = os.path.join(out_root, f"sigma_{_sigma_tag(sigma)}")
        s11_name = "s11.csv" if args.openems_s11_source == "zin" else "s11_wave.csv"
        s11_path = os.path.join(sigma_root, f"microstrip_{port_tag}_match", s11_name)
        if not os.path.exists(s11_path):
            raise FileNotFoundError(f"Missing openEMS s11.csv for match sigma {sigma}: {s11_path}")
        freq_src, s11_src = _load_openems_s11(s11_path, args.openems_s11_source)
        openems_matches.append(
            _mask_s11_mag(
                _interp_s11(freq_src, s11_src, np.asarray(freqs, dtype=np.float64)),
                args.s11_max_mag,
            )
        )

    from tests import test_microstrip_port

    fdtdx_open = test_microstrip_port._run_case(
        "open",
        freqs=freqs,
        source_offset_mm=args.fdtdx_source_offset_mm,
        port_offset_mm=args.fdtdx_port_offset_mm,
        port_len_mm=args.fdtdx_port_len_mm,
        resolution_mm=args.fdtdx_resolution_mm,
        i_mode=args.fdtdx_port_vi_mode,
    )
    fdtdx_short = test_microstrip_port._run_case(
        "short",
        freqs=freqs,
        source_offset_mm=args.fdtdx_source_offset_mm,
        port_offset_mm=args.fdtdx_port_offset_mm,
        port_len_mm=args.fdtdx_port_len_mm,
        resolution_mm=args.fdtdx_resolution_mm,
        i_mode=args.fdtdx_port_vi_mode,
    )
    v_open = None
    i_short = None
    if args.metric == "thevenin":
        v_open = _complex_from_case(fdtdx_open, "v")
        i_short = _complex_from_case(fdtdx_short, "i")
        s11_meas_list = []
        for case in (fdtdx_open, fdtdx_short):
            v_load = _complex_from_case(case, "v")
            _, s11 = calibration.thevenin_s11(v_load, v_open, i_short)
            s11_meas_list.append(s11)
        for sigma in cal_sigmas:
            match_case = test_microstrip_port._run_case(
                "match",
                freqs=freqs,
                load_sigma=float(sigma),
                source_offset_mm=args.fdtdx_source_offset_mm,
                port_offset_mm=args.fdtdx_port_offset_mm,
                port_len_mm=args.fdtdx_port_len_mm,
                resolution_mm=args.fdtdx_resolution_mm,
                i_mode=args.fdtdx_port_vi_mode,
            )
            v_load = _complex_from_case(match_case, "v")
            _, s11 = calibration.thevenin_s11(v_load, v_open, i_short)
            s11_meas_list.append(s11)
    else:
        s11_meas_list = [_s11_from_case(fdtdx_open, args.metric), _s11_from_case(fdtdx_short, args.metric)]
        for sigma in cal_sigmas:
            match_case = test_microstrip_port._run_case(
                "match",
                freqs=freqs,
                load_sigma=float(sigma),
                source_offset_mm=args.fdtdx_source_offset_mm,
                port_offset_mm=args.fdtdx_port_offset_mm,
                port_len_mm=args.fdtdx_port_len_mm,
                resolution_mm=args.fdtdx_resolution_mm,
                i_mode=args.fdtdx_port_vi_mode,
            )
            s11_meas_list.append(_s11_from_case(match_case, args.metric))

    s11_meas = _mask_s11_mag(np.vstack(s11_meas_list), args.s11_max_mag)
    s11_true = np.vstack([openems["open"], openems["short"], *openems_matches])

    A, B, C, valid = calibration.solve_oneport_calibration(s11_true, s11_meas)
    s11_cal = calibration.apply_oneport_calibration(s11_meas, A, B, C)
    err = np.abs(s11_cal - s11_true)
    err_db = 20.0 * np.log10(np.maximum(err, 1e-12))
    err_uncal = np.abs(s11_meas - s11_true)
    err_uncal_db = 20.0 * np.log10(np.maximum(err_uncal, 1e-12))
    err_mask = np.isfinite(err_db)
    err_mag_mask = np.isfinite(err)
    err_mag_rms = float(np.sqrt(np.nanmean(err[err_mag_mask] ** 2))) if np.any(err_mag_mask) else float("nan")
    err_mag_max = float(np.nanmax(err[err_mag_mask])) if np.any(err_mag_mask) else float("nan")
    err_rms = float(np.sqrt(np.nanmean(err_db[err_mask] ** 2))) if np.any(err_mask) else float("nan")
    err_max = float(np.nanmax(err_db[err_mask])) if np.any(err_mask) else float("nan")
    err_uncal_mask = np.isfinite(err_uncal_db)
    err_uncal_mag_mask = np.isfinite(err_uncal)
    err_uncal_mag_rms = (
        float(np.sqrt(np.nanmean(err_uncal[err_uncal_mag_mask] ** 2))) if np.any(err_uncal_mag_mask) else float("nan")
    )
    err_uncal_mag_max = float(np.nanmax(err_uncal[err_uncal_mag_mask])) if np.any(err_uncal_mag_mask) else float("nan")
    err_uncal_rms = (
        float(np.sqrt(np.nanmean(err_uncal_db[err_uncal_mask] ** 2))) if np.any(err_uncal_mask) else float("nan")
    )
    err_uncal_max = float(np.nanmax(err_uncal_db[err_uncal_mask])) if np.any(err_uncal_mask) else float("nan")

    payload = {
        "fixture": f"microstrip_{port_tag}",
        "metric": args.metric,
        "freq_hz": [float(f) for f in freqs],
        "a_real": [float(np.real(v)) for v in A],
        "a_imag": [float(np.imag(v)) for v in A],
        "b_real": [float(np.real(v)) for v in B],
        "b_imag": [float(np.imag(v)) for v in B],
        "c_real": [float(np.real(v)) for v in C],
        "c_imag": [float(np.imag(v)) for v in C],
        "valid_mask": [bool(v) for v in valid],
        "f0_hz": float(f0_hz),
        "valid_count": int(np.sum(valid)),
        "calibration_sigmas": [float(s) for s in cal_sigmas],
        "calibration_standards": ["open", "short"]
        + [f"match_sigma_{_sigma_tag(sigma)}" for sigma in cal_sigmas],
        "calibration_standard_count": int(s11_true.shape[0]),
        "error_uncal_db_rms": err_uncal_rms,
        "error_uncal_db_max": err_uncal_max,
        "error_db_rms": err_rms,
        "error_db_max": err_max,
        "error_uncal_mag_rms": err_uncal_mag_rms,
        "error_uncal_mag_max": err_uncal_mag_max,
        "error_mag_rms": err_mag_rms,
        "error_mag_max": err_mag_max,
    }

    if args.validate_sigma is not None:
        val_root = out_root + f"_val_{_sigma_tag(args.validate_sigma)}"
        _run_openems(
            args.openems_python,
            antenna_opt_root,
            val_root,
            fmin_hz,
            fmax_hz,
            f0_hz,
            len(freqs),
            load_sigma=float(args.validate_sigma),
            load_modes="match",
            port_types=args.openems_port_types,
            extra_args=extra_args,
        )
        s11_name = "s11.csv" if args.openems_s11_source == "zin" else "s11_wave.csv"
        s11_val_path = os.path.join(val_root, f"microstrip_{port_tag}_match", s11_name)
        freq_val, s11_val_true = _load_openems_s11(s11_val_path, args.openems_s11_source)
        s11_val_true = _interp_s11(freq_val, s11_val_true, np.asarray(freqs, dtype=np.float64))
        fdtdx_val = test_microstrip_port._run_case(
            "match",
            freqs=freqs,
            load_sigma=float(args.validate_sigma),
            source_offset_mm=args.fdtdx_source_offset_mm,
            port_offset_mm=args.fdtdx_port_offset_mm,
            port_len_mm=args.fdtdx_port_len_mm,
            resolution_mm=args.fdtdx_resolution_mm,
            i_mode=args.fdtdx_port_vi_mode,
        )
        if args.metric == "thevenin":
            if v_open is None or i_short is None:
                raise ValueError("thevenin validation requires open/short references")
            v_load = _complex_from_case(fdtdx_val, "v")
            _, s11_val_meas = calibration.thevenin_s11(v_load, v_open, i_short)
        else:
            s11_val_meas = _s11_from_case(fdtdx_val, args.metric)
        s11_val_cal = calibration.apply_oneport_calibration(s11_val_meas, A, B, C)
        err_val = np.abs(s11_val_cal - s11_val_true)
        err_val_db = 20.0 * np.log10(np.maximum(err_val, 1e-12))
        err_val_uncal = np.abs(s11_val_meas - s11_val_true)
        err_val_uncal_db = 20.0 * np.log10(np.maximum(err_val_uncal, 1e-12))
        mask_val = np.isfinite(err_val_db)
        mask_val_uncal = np.isfinite(err_val_uncal_db)
        mask_val_mag = np.isfinite(err_val)
        mask_val_uncal_mag = np.isfinite(err_val_uncal)
        val_in_cal = any(abs(s - float(args.validate_sigma)) < 1e-9 for s in cal_sigmas)
        payload.update(
            {
                "validation_sigma": float(args.validate_sigma),
                "validation_sigma_in_calibration": bool(val_in_cal),
                "validation_error_db_rms": float(np.sqrt(np.nanmean(err_val_db[mask_val] ** 2)))
                if np.any(mask_val)
                else float("nan"),
                "validation_error_db_max": float(np.nanmax(err_val_db[mask_val])) if np.any(mask_val) else float("nan"),
                "validation_error_uncal_db_rms": float(np.sqrt(np.nanmean(err_val_uncal_db[mask_val_uncal] ** 2)))
                if np.any(mask_val_uncal)
                else float("nan"),
                "validation_error_uncal_db_max": float(np.nanmax(err_val_uncal_db[mask_val_uncal]))
                if np.any(mask_val_uncal)
                else float("nan"),
                "validation_error_mag_rms": float(np.sqrt(np.nanmean(err_val[mask_val_mag] ** 2)))
                if np.any(mask_val_mag)
                else float("nan"),
                "validation_error_mag_max": float(np.nanmax(err_val[mask_val_mag])) if np.any(mask_val_mag) else float("nan"),
                "validation_error_uncal_mag_rms": float(np.sqrt(np.nanmean(err_val_uncal[mask_val_uncal_mag] ** 2)))
                if np.any(mask_val_uncal_mag)
                else float("nan"),
                "validation_error_uncal_mag_max": float(np.nanmax(err_val_uncal[mask_val_uncal_mag]))
                if np.any(mask_val_uncal_mag)
                else float("nan"),
            }
        )

    out_path = args.out or os.path.join(repo_root, "antenna_fdtdx", "calibration", "port_calibration.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    calibration.save_calibration(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

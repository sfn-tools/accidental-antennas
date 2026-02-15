"""Baseline cross-checks for FDTDX port extraction vs openEMS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

from sim import calibration
from sim import metrics

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _fdtdx_microstrip() -> Dict[str, Dict]:
    from tests import test_microstrip_port

    results = {}
    for load_mode in ("open", "short", "match"):
        metrics = test_microstrip_port._run_case(load_mode)
        results[load_mode] = metrics
    _add_thevenin_metrics(results, label="microstrip")
    return results


def _fdtdx_plate() -> Dict[str, Dict]:
    from tests import test_port_vi

    results = {}
    cases = {
        "open": (None, False),
        "short": (5.0e6, False),
        "match": (2.5, False),
    }
    for name, (sigma, load_is_pec) in cases.items():
        metrics = test_port_vi._run_case(load_sigma=sigma, load_is_pec=load_is_pec)
        results[name] = metrics
    _add_thevenin_metrics(results, label="plate")
    return results


def _run_openems_suite(
    openems_python: str,
    antenna_opt_root: str,
    out_root: str,
    summary_path: str,
    port_types: str,
    load_modes: str,
    case: str,
    extra_args: List[str] | None,
) -> Tuple[int, str, str]:
    cmd = [
        openems_python,
        "-m",
        "tools.baseline_suite",
        "--case",
        case,
        "--port-types",
        port_types,
        "--load-modes",
        load_modes,
        "--out-root",
        out_root,
        "--summary",
        summary_path,
        "--prune",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        cwd=antenna_opt_root,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _complex_from_case(case: Dict, prefix: str) -> np.ndarray:
    real_key = f"{prefix}_real"
    imag_key = f"{prefix}_imag"
    if real_key not in case or imag_key not in case:
        raise KeyError(f"Missing {real_key}/{imag_key}")
    return np.asarray(case[real_key], dtype=np.float64) + 1j * np.asarray(case[imag_key], dtype=np.float64)


def _add_thevenin_metrics(results: Dict[str, Dict], label: str) -> None:
    try:
        open_case = results["open"]
        short_case = results["short"]
    except KeyError:
        results["_thevenin_error"] = f"{label}:missing_open_short"
        return
    try:
        v_open = _complex_from_case(open_case, "v")
        i_short = _complex_from_case(short_case, "i")
    except KeyError as exc:
        results["_thevenin_error"] = f"{label}:missing_vi:{exc}"
        return

    freq_open = np.asarray(open_case.get("freq_hz", []), dtype=np.float64)
    freq_short = np.asarray(short_case.get("freq_hz", []), dtype=np.float64)
    if freq_open.size == 0 or freq_short.size == 0:
        results["_thevenin_error"] = f"{label}:missing_freq"
        return
    if freq_open.shape != freq_short.shape or not np.allclose(freq_open, freq_short):
        results["_thevenin_error"] = f"{label}:freq_mismatch_open_short"
        return

    for name, case in results.items():
        if not isinstance(case, dict):
            continue
        if "v_real" not in case or "v_imag" not in case:
            case["thevenin_error"] = f"{label}:{name}:missing_v"
            continue
        freq_case = np.asarray(case.get("freq_hz", []), dtype=np.float64)
        if freq_case.size == 0 or freq_case.shape != freq_open.shape or not np.allclose(freq_case, freq_open):
            case["thevenin_error"] = f"{label}:{name}:freq_mismatch"
            continue
        v_load = _complex_from_case(case, "v")
        z_load, s11 = calibration.thevenin_s11(v_load, v_open, i_short)
        rl_db = -metrics.s11_db(s11)
        f0 = float(case.get("f0_hz", np.nan))
        if np.isfinite(f0):
            f0_idx = int(np.argmin(np.abs(freq_case - f0)))
        else:
            f0_idx = 0
        case["s11_thevenin_real"] = [float(np.real(val)) for val in s11]
        case["s11_thevenin_imag"] = [float(np.imag(val)) for val in s11]
        case["rl_thevenin_db_sweep"] = [float(val) for val in rl_db]
        case["rl_thevenin_db"] = float(rl_db[f0_idx]) if rl_db.size else float("nan")
        if z_load.size:
            case["zin_thevenin_real_ohm"] = float(np.real(z_load[f0_idx]))
            case["zin_thevenin_imag_ohm"] = float(np.imag(z_load[f0_idx]))
        else:
            case["zin_thevenin_real_ohm"] = float("nan")
            case["zin_thevenin_imag_ohm"] = float("nan")


def _apply_calibration(results: Dict[str, Dict], metric: str, cal_data: Dict) -> None:
    for name, case in results.items():
        if not isinstance(case, dict):
            continue
        try:
            s11_meas = _complex_from_case(case, f"s11_{metric}")
        except KeyError as exc:
            case["calibration_error"] = f"missing_{metric}:{exc}"
            continue
        freq = np.asarray(case.get("freq_hz", []), dtype=np.float64)
        if freq.size == 0:
            case["calibration_error"] = "missing_freq"
            continue
        cal_interp = calibration.interp_calibration(cal_data, freq)
        s11_cal = calibration.apply_oneport_calibration(s11_meas, cal_interp["A"], cal_interp["B"], cal_interp["C"])
        rl_db = -metrics.s11_db(s11_cal)
        f0 = float(case.get("f0_hz", np.nan))
        f0_idx = int(np.argmin(np.abs(freq - f0))) if np.isfinite(f0) else 0
        case["s11_cal_real"] = [float(np.real(val)) for val in s11_cal]
        case["s11_cal_imag"] = [float(np.imag(val)) for val in s11_cal]
        case["rl_cal_db_sweep"] = [float(val) for val in rl_db]
        case["rl_cal_db"] = float(rl_db[f0_idx]) if rl_db.size else float("nan")
        case["calibration_metric"] = metric


def _compare_results(
    fdtdx: Dict[str, Dict],
    openems: Dict[str, Dict],
    port_types: List[str],
    load_modes: List[str],
    rl_mode: str,
    cal_metric: str | None,
) -> Dict[str, Dict]:
    comparison = {}
    for port_type in port_types:
        for load_mode in load_modes:
            key = f"microstrip_{port_type}_{load_mode}"
            o = openems.get(key)
            if o is None:
                comparison[key] = {"error": "missing_openems_case"}
                continue
            f = fdtdx.get(load_mode)
            if f is None:
                comparison[key] = {"error": "missing_fdtdx_case"}
                continue
            use_cal = cal_metric is not None and rl_mode == cal_metric
            rl_key = "rl_cal_db" if use_cal else f"rl_{rl_mode}_db"
            zin_real_key = f"zin_{rl_mode}_real_ohm"
            zin_imag_key = f"zin_{rl_mode}_imag_ohm"
            rl_fdtdx = f.get(rl_key)
            zin_real_fdtdx = f.get(zin_real_key)
            zin_imag_fdtdx = f.get(zin_imag_key)
            comparison[key] = {
                "rl_f0_db_fdtdx": rl_fdtdx,
                "rl_f0_db_openems": o.get("rl_f0_db"),
                "rl_delta_db": (
                    float(o.get("rl_f0_db")) - float(rl_fdtdx)
                    if o.get("rl_f0_db") is not None and rl_fdtdx is not None
                    else None
                ),
                "zin_real_fdtdx": zin_real_fdtdx,
                "zin_real_openems": o.get("zin_f0_real_ohm"),
                "zin_imag_fdtdx": zin_imag_fdtdx,
                "zin_imag_openems": o.get("zin_f0_imag_ohm"),
                "rl_mode_used": f"{rl_mode}_cal" if use_cal else rl_mode,
            }
    for load_mode in ("open", "short", "match"):
        key = f"plate_{load_mode}"
        o = openems.get(key)
        if o is None:
            continue
        f = fdtdx.get("plate", {}).get(load_mode)
        if f is None:
            comparison[key] = {"error": "missing_fdtdx_case"}
            continue
        plate_mode = "vi" if rl_mode == "thevenin" else rl_mode
        rl_key = f"rl_{plate_mode}_db"
        zin_real_key = f"zin_{plate_mode}_real_ohm"
        zin_imag_key = f"zin_{plate_mode}_imag_ohm"
        rl_fdtdx = f.get(rl_key)
        zin_real_fdtdx = f.get(zin_real_key)
        zin_imag_fdtdx = f.get(zin_imag_key)
        comparison[key] = {
            "rl_f0_db_fdtdx": rl_fdtdx,
            "rl_f0_db_openems": o.get("rl_f0_db"),
            "rl_delta_db": (
                float(o.get("rl_f0_db")) - float(rl_fdtdx)
                if o.get("rl_f0_db") is not None and rl_fdtdx is not None
                else None
            ),
            "zin_real_fdtdx": zin_real_fdtdx,
            "zin_real_openems": o.get("zin_f0_real_ohm"),
            "zin_imag_fdtdx": zin_imag_fdtdx,
            "zin_imag_openems": o.get("zin_f0_imag_ohm"),
            "rl_mode_used": plate_mode,
        }
    return comparison


def _save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openems-python", default="python3", help="Python for openEMS")
    parser.add_argument("--openems-root", default=None, help="Path to antenna_opt")
    parser.add_argument("--out-root", default=None, help="Output directory for openEMS runs")
    parser.add_argument("--summary", default=None, help="Write summary JSON here")
    parser.add_argument("--skip-openems", action="store_true", help="Skip openEMS runs")
    parser.add_argument("--port-types", default="lumped", help="Comma-separated port types")
    parser.add_argument("--load-modes", default="open,short,match", help="Comma-separated load modes")
    parser.add_argument("--rl-mode", default="flux", help="Use vi or flux metrics for comparisons")
    parser.add_argument("--port-calibration", default=None, help="Calibration JSON for port S11")
    parser.add_argument("--port-calibration-metric", choices=["vi", "flux", "thevenin"], default=None)
    parser.add_argument("--openems-args", default=None, help="Extra args passed to antenna_opt.tools.baseline_suite")
    args = parser.parse_args()
    port_types = [p.strip() for p in args.port_types.split(",") if p.strip()]
    load_modes = [p.strip() for p in args.load_modes.split(",") if p.strip()]
    rl_mode = args.rl_mode.strip().lower()
    if rl_mode not in {"vi", "flux", "thevenin"}:
        raise ValueError(f"Unknown rl-mode: {args.rl_mode}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    antenna_opt_root = args.openems_root or os.path.join(repo_root, "antenna_opt")
    out_root = args.out_root or os.path.join(antenna_opt_root, "reports", "baseline_suite")
    summary_path = args.summary or os.path.join(repo_root, "antenna_fdtdx", "tmp", "baseline_suite.json")

    payload: Dict[str, Dict] = {"fdtdx": {}, "openems": {}, "comparison": {}}
    try:
        payload["fdtdx"] = _fdtdx_microstrip()
        payload["fdtdx"]["plate"] = _fdtdx_plate()
    except Exception as exc:
        payload["fdtdx_error"] = str(exc)

    cal_metric = None
    if payload.get("fdtdx") and args.port_calibration:
        cal_metric = args.port_calibration_metric or (rl_mode if rl_mode in {"vi", "flux", "thevenin"} else "flux")
        cal_data = calibration.load_calibration(args.port_calibration)
        _apply_calibration(payload["fdtdx"], cal_metric, cal_data)
        payload["fdtdx_calibration"] = {
            "path": args.port_calibration,
            "metric": cal_metric,
        }

    if not args.skip_openems:
        extra_args = args.openems_args.split() if args.openems_args else None
        microstrip_summary = os.path.join(out_root, "summary_microstrip.json")
        rc_micro, out_micro, err_micro = _run_openems_suite(
            args.openems_python,
            antenna_opt_root,
            out_root,
            microstrip_summary,
            port_types=",".join(port_types),
            load_modes=",".join(load_modes),
            case="microstrip",
            extra_args=extra_args,
        )
        plate_summary = os.path.join(out_root, "summary_plate.json")
        rc_plate, out_plate, err_plate = _run_openems_suite(
            args.openems_python,
            antenna_opt_root,
            out_root,
            plate_summary,
            port_types="lumped",
            load_modes=",".join(load_modes),
            case="plate",
            extra_args=extra_args,
        )
        payload["openems_cmd"] = {
            "microstrip": {
                "returncode": rc_micro,
                "stdout": out_micro.strip(),
                "stderr": err_micro.strip(),
            },
            "plate": {
                "returncode": rc_plate,
                "stdout": out_plate.strip(),
                "stderr": err_plate.strip(),
            },
        }
        if rc_micro == 0 and os.path.exists(microstrip_summary):
            with open(microstrip_summary, "r", encoding="utf-8") as handle:
                payload["openems"].update(json.load(handle))
        if rc_plate == 0 and os.path.exists(plate_summary):
            with open(plate_summary, "r", encoding="utf-8") as handle:
                payload["openems"].update(json.load(handle))
        if not payload.get("openems"):
            payload["openems_error"] = "openems_suite_failed"

    if payload.get("fdtdx") and payload.get("openems"):
        payload["comparison"] = _compare_results(
            payload["fdtdx"],
            payload["openems"],
            port_types,
            load_modes,
            rl_mode,
            cal_metric,
        )

    _save_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

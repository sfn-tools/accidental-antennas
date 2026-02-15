"""Periodic CPU threshold evaluation for in-flight topology runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, Optional

import numpy as np


def _read_last_epoch(history_path: str) -> Optional[int]:
    if not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None


def _load_model_name(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    model = cfg.get("model")
    if not model:
        raise ValueError(f"Missing model in {cfg_path}")
    return str(model)


def _threshold_params(src_path: str, dst_path: str, threshold: float) -> float:
    params = np.load(src_path)
    params = np.where(params >= threshold, 1.0, 0.0).astype(np.float32)
    np.save(dst_path, params)
    return float(params.mean())


def _run_cpu_eval(
    model: str,
    params_path: str,
    quality: str,
    run_root: str,
    rl_mode: str,
    port_reference: Optional[str],
    port_calibration: Optional[str],
    calibration_metric: Optional[str],
    params_resample: bool,
    resample_method: str,
) -> str:
    if rl_mode == "thevenin" and not port_reference:
        print(
            "[threshold_watch] rl_mode=thevenin requires --port-reference; falling back to vi",
            file=sys.stderr,
        )
        rl_mode = "vi"
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLUGINS"] = "cpu"
    env["JAX_PJRT_BACKEND"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cmd = [
        sys.executable,
        "-m",
        "sim.run_one",
        "--model",
        model,
        "--quality",
        quality,
        "--backend",
        "cpu",
        "--rl-mode",
        rl_mode,
        "--params",
        params_path,
        "--run-root",
        run_root,
        "--force",
    ]
    if params_resample:
        cmd.extend(["--params-resample", "--params-resample-method", resample_method])
    if port_reference:
        cmd.extend(["--port-reference", port_reference])
    if port_calibration:
        cmd.extend(["--port-calibration", port_calibration])
    if calibration_metric:
        cmd.extend(["--port-calibration-metric", calibration_metric])
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    out = proc.stdout + "\n" + proc.stderr
    match = re.search(r"Run output directory:\s*(\S+)", out)
    if not match:
        raise RuntimeError(f"Failed to locate run directory in output:\n{out}")
    return match.group(1)


def _write_eval(run_dir: str, entry: Dict) -> None:
    out_path = os.path.join(run_dir, "threshold_eval_cpu.jsonl")
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _evaluate_once(
    run_dir: str,
    model: str,
    epoch: int,
    threshold: float,
    quality: str,
    run_root: str,
    rl_mode: str,
    port_reference: Optional[str],
    port_calibration: Optional[str],
    calibration_metric: Optional[str],
    params_resample: bool,
    resample_method: str,
) -> None:
    tag = f"thr{threshold:.2f}".replace(".", "p")
    params_path = os.path.join(run_dir, f"best_{tag}_params.npy")
    if not os.path.exists(params_path):
        params_path = os.path.join(run_dir, "best_rl_params.npy")
    if not os.path.exists(params_path):
        params_path = os.path.join(run_dir, "best_params.npy")
    if not os.path.exists(params_path):
        params_path = os.path.join(run_dir, "last_params.npy")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params found in {run_dir}")
    thr_path = os.path.join(run_dir, f"params_{tag}_e{epoch + 1:04d}.npy")
    fill_frac = _threshold_params(params_path, thr_path, threshold)
    out_dir = _run_cpu_eval(
        model,
        thr_path,
        quality,
        run_root,
        rl_mode,
        port_reference,
        port_calibration,
        calibration_metric,
        params_resample,
        resample_method,
    )
    metrics_path = os.path.join(run_root, out_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    rl_min = metrics.get("rl_min_in_band_db")
    rl_peak = metrics.get("rl_peak_in_band_db")
    f_peak = metrics.get("f_peak_in_band_hz")
    rl_min_matched = metrics.get("rl_min_in_band_matched_db")
    rl_min_match3 = metrics.get("match3_rl_min_in_band_db")

    valid_flag = metrics.get("valid")
    if valid_flag is None:
        valid_flag = True
    entry = {
        "epoch": epoch,
        "threshold": threshold,
        "quality": quality,
        "run_dir": out_dir,
        "rl_mode": rl_mode,
        "valid": bool(valid_flag),
        "error": metrics.get("error"),
        "rl_min_db": rl_min,
        "rl_min_flux_db": metrics.get("rl_min_in_band_flux_db"),
        "rl_min_cal_db": metrics.get("rl_min_in_band_cal_db"),
        "rl_min_matched_db": rl_min_matched,
        "rl_peak_db": rl_peak,
        "f_peak_hz": f_peak,
        "fb_db": metrics.get("fb_db"),
        "eta_rad": metrics.get("eta_rad"),
        "zin_f0_real_ohm": metrics.get("zin_f0_real_ohm"),
        "zin_f0_imag_ohm": metrics.get("zin_f0_imag_ohm"),
        "match_bandwidth_hz": metrics.get("match_bandwidth_hz"),
        "match_bandwidth_frac": metrics.get("match_bandwidth_frac"),
        "match3_rl_min_in_band_db": rl_min_match3,
        "match3_bandwidth_hz": metrics.get("match3_bandwidth_hz"),
        "match3_bandwidth_frac": metrics.get("match3_bandwidth_frac"),
        "fill_frac": fill_frac,
    }
    _write_eval(run_dir, entry)
    print(
        f"[{os.path.basename(run_dir)}] epoch {epoch} thr={threshold:.2f} "
        f"rl_min={entry['rl_min_db']} rl_peak={entry['rl_peak_db']} run={out_dir}"
    )
    return entry, out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="opt_runs/<hash> directory (repeatable)",
    )
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quality", default="coarse", choices=["coarse", "mid", "medium", "fine", "high", "fast"])
    parser.add_argument("--rl-mode", choices=["vi", "flux", "thevenin"], default="flux")
    parser.add_argument("--port-reference", default=None)
    parser.add_argument("--port-calibration", default=None)
    parser.add_argument("--port-calibration-metric", default=None)
    parser.add_argument("--params-resample", action="store_true")
    parser.add_argument("--params-resample-method", choices=["nearest", "linear"], default="nearest")
    parser.add_argument("--sleep", type=float, default=120.0)
    parser.add_argument("--run-root", default=None, help="Output directory for CPU eval runs")
    args = parser.parse_args()

    run_root = args.run_root
    if run_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        run_root = os.path.join(repo_root, "runs_threshold")
    os.makedirs(run_root, exist_ok=True)

    interval = max(1, int(args.interval))
    state: Dict[str, int] = {}
    models: Dict[str, str] = {}

    while True:
        for run_dir in args.run:
            run_dir = os.path.abspath(run_dir)
            history_path = os.path.join(run_dir, "history.jsonl")
            epoch = _read_last_epoch(history_path)
            if epoch is None:
                continue
            last_eval = state.get(run_dir, -1)
            if epoch <= last_eval:
                continue
            if (epoch + 1) % interval != 0:
                continue
            model = models.get(run_dir)
            if model is None:
                model = _load_model_name(run_dir)
                models[run_dir] = model
            _evaluate_once(
                run_dir,
                model,
                epoch,
                args.threshold,
                args.quality,
                run_root,
                args.rl_mode,
                args.port_reference,
                args.port_calibration,
                args.port_calibration_metric,
                args.params_resample,
                args.params_resample_method,
            )
            state[run_dir] = epoch
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())

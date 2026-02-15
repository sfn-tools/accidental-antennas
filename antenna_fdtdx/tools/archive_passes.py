"""Archive passing antenna designs with openEMS + FDTDX artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _passes(metrics: Dict, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    failures = []
    rl = _maybe_float(metrics.get("rl_min_in_band_db"))
    bw_frac = None
    if args.use_match3 and metrics.get("match3_rl_min_in_band_db") is not None:
        rl = _maybe_float(metrics.get("match3_rl_min_in_band_db"))
        bw_frac = _maybe_float(metrics.get("match3_bandwidth_frac"))
    elif args.use_matched_rl and metrics.get("rl_min_in_band_matched_db") is not None:
        rl = _maybe_float(metrics.get("rl_min_in_band_matched_db"))
        bw_frac = _maybe_float(metrics.get("match_bandwidth_frac"))
    fb = _maybe_float(metrics.get("plane_fb_peak_db"))
    if fb is None:
        fb = _maybe_float(metrics.get("fb_db"))
    gain = _maybe_float(metrics.get("plane_gain_peak_realized_db"))
    if gain is None:
        gain = _maybe_float(metrics.get("gain_fwd_realized_db"))
    zin_real = _maybe_float(metrics.get("zin_f0_real_ohm"))
    zin_imag = _maybe_float(metrics.get("zin_f0_imag_ohm"))
    p_acc = _maybe_float(metrics.get("p_acc_w"))
    rad_eff = _maybe_float(metrics.get("rad_eff"))

    if rl is None or rl < args.min_rl_db:
        failures.append(f"rl_min<{args.min_rl_db}")
    if args.min_bw_frac is not None:
        if bw_frac is None or bw_frac < args.min_bw_frac:
            failures.append(f"bw_frac<{args.min_bw_frac}")
    if fb is None or fb < args.min_fb_db:
        failures.append(f"fb<{args.min_fb_db}")
    if gain is None or gain < args.min_gain_db:
        failures.append(f"gain<{args.min_gain_db}")
    if zin_real is None or zin_real < args.min_zin_real or zin_real > args.max_zin_real:
        failures.append(f"zin_real_not_in[{args.min_zin_real},{args.max_zin_real}]")
    if zin_imag is None or abs(zin_imag) > args.max_zin_imag:
        failures.append(f"zin_imag_abs>{args.max_zin_imag}")
    if p_acc is None or p_acc < args.min_pacc_w:
        failures.append(f"p_acc<{args.min_pacc_w}")
    if rad_eff is None or rad_eff < args.min_rad_eff or rad_eff > args.max_rad_eff:
        failures.append(f"rad_eff_not_in[{args.min_rad_eff},{args.max_rad_eff}]")

    return len(failures) == 0, failures


def _safe_copy(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _safe_symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def _collect_openems_runs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_openems_root = os.path.abspath(os.path.join(repo_root, "..", "antenna_opt", "runs_fdtdx"))
    default_out_root = os.path.join(repo_root, "passing_designs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--openems-root", default=default_openems_root)
    parser.add_argument("--out-root", default=default_out_root)
    parser.add_argument("--min-rl-db", type=float, default=10.0)
    parser.add_argument("--min-fb-db", type=float, default=10.0)
    parser.add_argument("--min-gain-db", type=float, default=5.0)
    parser.add_argument("--min-zin-real", type=float, default=35.0)
    parser.add_argument("--max-zin-real", type=float, default=65.0)
    parser.add_argument("--max-zin-imag", type=float, default=10.0)
    parser.add_argument("--min-pacc-w", type=float, default=1e-6)
    parser.add_argument("--min-rad-eff", type=float, default=0.0)
    parser.add_argument("--max-rad-eff", type=float, default=1.2)
    parser.add_argument("--use-matched-rl", action="store_true")
    parser.add_argument("--use-match3", action="store_true", help="Prefer 3-element match metrics if available")
    parser.add_argument("--min-bw-frac", type=float, default=None)
    args = parser.parse_args()
    print(f"[archive_passes] openems_root={args.openems_root}")
    print(f"[archive_passes] out_root={args.out_root}")

    openems_runs = _collect_openems_runs(args.openems_root)
    if not openems_runs:
        print("No openEMS runs found.")
        return 0

    os.makedirs(args.out_root, exist_ok=True)
    archived = 0
    for run_dir in sorted(openems_runs):
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        metrics = _load_json(metrics_path)
        ok, failures = _passes(metrics, args)
        if not ok:
            continue

        meta_path = os.path.join(run_dir, "fdtdx_meta.json")
        if not os.path.exists(meta_path):
            continue
        meta = _load_json(meta_path)
        fdtdx_run = meta.get("fdtdx_run")
        if not fdtdx_run or not os.path.isdir(fdtdx_run):
            continue

        model = "unknown"
        try:
            fdtdx_params = _load_json(os.path.join(fdtdx_run, "params.json"))
            model = fdtdx_params.get("model", model)
        except Exception:
            model = model

        run_id = os.path.basename(run_dir)
        tag = f"{model}__{run_id}"
        dest = os.path.join(args.out_root, tag)
        if os.path.exists(dest):
            continue

        os.makedirs(dest, exist_ok=True)
        _safe_symlink(run_dir, os.path.join(dest, "openems_run"))
        _safe_symlink(fdtdx_run, os.path.join(dest, "fdtdx_run"))

        _safe_copy(metrics_path, os.path.join(dest, "openems_metrics.json"))
        _safe_copy(os.path.join(run_dir, "s11.csv"), os.path.join(dest, "openems_s11.csv"))
        _safe_copy(os.path.join(run_dir, "s11.png"), os.path.join(dest, "openems_s11.png"))
        _safe_copy(os.path.join(run_dir, "s11_matched.csv"), os.path.join(dest, "openems_s11_matched.csv"))
        _safe_copy(os.path.join(run_dir, "s11_matched.png"), os.path.join(dest, "openems_s11_matched.png"))
        _safe_copy(os.path.join(run_dir, "s11_matched_smith.csv"), os.path.join(dest, "openems_s11_matched_smith.csv"))
        _safe_copy(os.path.join(run_dir, "s11_matched3.csv"), os.path.join(dest, "openems_s11_matched3.csv"))
        _safe_copy(os.path.join(run_dir, "s11_matched3.png"), os.path.join(dest, "openems_s11_matched3.png"))
        _safe_copy(os.path.join(run_dir, "s11_matched3_smith.csv"), os.path.join(dest, "openems_s11_matched3_smith.csv"))
        _safe_copy(os.path.join(run_dir, "fdtdx_meta.json"), os.path.join(dest, "openems_fdtdx_meta.json"))
        _safe_copy(os.path.join(run_dir, "meta.json"), os.path.join(dest, "openems_meta.json"))
        _safe_copy(os.path.join(run_dir, "pattern_3d.vtk"), os.path.join(dest, "pattern_3d.vtk"))
        _safe_copy(os.path.join(run_dir, "copper_top.svg"), os.path.join(dest, "copper_top.svg"))
        _safe_copy(os.path.join(run_dir, "copper_ground.svg"), os.path.join(dest, "copper_ground.svg"))

        _safe_copy(os.path.join(fdtdx_run, "geometry.npz"), os.path.join(dest, "fdtdx_geometry.npz"))
        _safe_copy(os.path.join(fdtdx_run, "params.json"), os.path.join(dest, "fdtdx_params.json"))
        _safe_copy(os.path.join(fdtdx_run, "metrics.json"), os.path.join(dest, "fdtdx_metrics.json"))
        _safe_copy(os.path.join(fdtdx_run, "s11.csv"), os.path.join(dest, "fdtdx_s11.csv"))
        _safe_copy(os.path.join(fdtdx_run, "s11.png"), os.path.join(dest, "fdtdx_s11.png"))
        _safe_copy(os.path.join(fdtdx_run, "s11_matched.csv"), os.path.join(dest, "fdtdx_s11_matched.csv"))
        _safe_copy(os.path.join(fdtdx_run, "s11_matched.png"), os.path.join(dest, "fdtdx_s11_matched.png"))
        _safe_copy(os.path.join(fdtdx_run, "s11_matched_smith.csv"), os.path.join(dest, "fdtdx_s11_matched_smith.csv"))
        for svg_path in glob.glob(os.path.join(fdtdx_run, "top_copper_thr*.svg")):
            _safe_copy(svg_path, os.path.join(dest, os.path.basename(svg_path)))

        manifest = {
            "openems_run": run_dir,
            "fdtdx_run": fdtdx_run,
            "model": model,
            "metrics": metrics,
            "filters": {
                "min_rl_db": args.min_rl_db,
                "min_fb_db": args.min_fb_db,
                "min_gain_db": args.min_gain_db,
                "min_zin_real": args.min_zin_real,
                "max_zin_real": args.max_zin_real,
                "max_zin_imag": args.max_zin_imag,
                "min_pacc_w": args.min_pacc_w,
                "min_rad_eff": args.min_rad_eff,
                "max_rad_eff": args.max_rad_eff,
                "use_matched_rl": args.use_matched_rl,
                "use_match3": args.use_match3,
                "min_bw_frac": args.min_bw_frac,
            },
        }
        with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        archived += 1
        print(f"ARCHIVED {tag}")

    if archived == 0:
        print("No passing designs found.")
    else:
        print(f"Archived {archived} design(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

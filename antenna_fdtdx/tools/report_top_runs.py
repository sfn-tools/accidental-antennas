#!/usr/bin/env python3
"""
Summarize top FDTDX runs and map them to OpenEMS validations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _parse_since(value: Optional[str], repo_root: str) -> float:
    if not value:
        return 0.0
    raw = value.strip()
    if not raw:
        return 0.0
    if raw.isdigit():
        return float(raw)
    if raw.startswith("commit:"):
        commit = raw.split(":", 1)[1].strip()
        if commit:
            return _commit_ts(commit, repo_root)
    # Try ISO format
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass
    # Try common format
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            continue
    raise ValueError(f"Could not parse --since value: {value}")


def _commit_ts(commit: str, repo_root: str) -> float:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", commit],
            cwd=repo_root,
            text=True,
        ).strip()
        return float(out)
    except Exception as exc:
        raise ValueError(f"Failed to resolve commit {commit}: {exc}") from exc


def _history_stats(history: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [h.get(key) for h in history if isinstance(h.get(key), (int, float))]
    return max(vals) if vals else None


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return f"{val:.2f}"


def _load_history(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
        return [json.loads(line) for line in lines]
    except Exception:
        return []


def _collect_fdtdx_runs(
    opt_root: str,
    since_ts: float,
    min_epochs: int,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not os.path.isdir(opt_root):
        return runs
    for name in os.listdir(opt_root):
        run_dir = os.path.join(opt_root, name)
        cfg_path = os.path.join(run_dir, "config.json")
        hist_path = os.path.join(run_dir, "history.jsonl")
        if not os.path.exists(cfg_path) or not os.path.exists(hist_path):
            continue
        if os.path.getmtime(cfg_path) < since_ts:
            continue
        cfg = _load_json(cfg_path)
        if not cfg:
            continue
        history = _load_history(hist_path)
        if len(history) < min_epochs:
            continue
        last = history[-1] if history else {}
        runs.append(
            {
                "run_id": name,
                "opt_run": run_dir,
                "model": cfg.get("model"),
                "seed": cfg.get("seed"),
                "init_mode": cfg.get("init_mode"),
                "epochs": len(history),
                "last": last,
                "best_rl_min_db": _history_stats(history, "rl_min_db"),
                "best_fb_db": _history_stats(history, "fb_db"),
                "best_fwd_eff_db": _history_stats(history, "fwd_eff_db"),
                "best_eta_rad": _history_stats(history, "eta_rad"),
            }
        )
    return runs


def _load_cpu_index(state_path: str) -> Dict[str, Dict[str, Any]]:
    state = _load_json(state_path) or {}
    return state.get("cpu_run_index", {}) or {}


def _choose_latest_cpu_run(cpu_dirs: List[str]) -> Optional[str]:
    if not cpu_dirs:
        return None
    def _mtime(path: str) -> float:
        metrics = os.path.join(path, "metrics.json")
        if os.path.exists(metrics):
            return os.path.getmtime(metrics)
        return os.path.getmtime(path)
    return max(cpu_dirs, key=_mtime)


def _load_openems_mapping(openems_root: str) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    mapping: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    if not os.path.isdir(openems_root):
        return mapping
    for name in os.listdir(openems_root):
        run_dir = os.path.join(openems_root, name)
        meta_path = os.path.join(run_dir, "fdtdx_meta.json")
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(meta_path) or not os.path.exists(metrics_path):
            continue
        meta = _load_json(meta_path)
        metrics = _load_json(metrics_path)
        if not meta or not metrics:
            continue
        fdtdx_run = meta.get("fdtdx_run")
        if fdtdx_run:
            mapping[fdtdx_run] = (name, metrics)
    return mapping


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_opt_root = os.path.join(repo_root, "opt_runs")
    default_openems_root = os.path.abspath(os.path.join(repo_root, "..", "antenna_opt", "runs_fdtdx"))
    default_campaign_state = os.path.join(repo_root, "campaign_state.json")

    parser = argparse.ArgumentParser(description="Report top FDTDX runs and OpenEMS mappings.")
    parser.add_argument("--since", default=None, help="Timestamp (epoch), ISO string, or 'commit:<hash>'")
    parser.add_argument("--top", type=int, default=5, help="Number of top FDTDX runs to show.")
    parser.add_argument("--min-epochs", type=int, default=1, help="Minimum epochs required.")
    parser.add_argument("--opt-root", default=default_opt_root)
    parser.add_argument("--openems-root", default=default_openems_root)
    parser.add_argument("--campaign-state", default=default_campaign_state)
    args = parser.parse_args()

    since_ts = _parse_since(args.since, repo_root)

    fdtdx_runs = _collect_fdtdx_runs(args.opt_root, since_ts, args.min_epochs)
    cpu_index = _load_cpu_index(args.campaign_state)
    opt_to_cpu: Dict[str, List[str]] = {}
    for cpu_dir, info in cpu_index.items():
        opt_run = info.get("opt_run")
        if opt_run:
            opt_to_cpu.setdefault(opt_run, []).append(cpu_dir)

    openems_map = _load_openems_mapping(args.openems_root)

    fdtdx_sorted = sorted(
        fdtdx_runs,
        key=lambda r: (r["best_rl_min_db"] if r["best_rl_min_db"] is not None else -1),
        reverse=True,
    )
    top_runs = fdtdx_sorted[: max(args.top, 0)]

    print(f"FDTDX runs since {datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()} : {len(fdtdx_runs)}")
    print(f"OpenEMS runs indexed: {len(openems_map)}")
    print("")
    print(f"Top {len(top_runs)} FDTDX runs by best rl_min_db:")
    for run in top_runs:
        last = run["last"]
        print(f"- {run['run_id']} {run['model']} seed{run['seed']} {run['init_mode']} epochs={run['epochs']}")
        print(
            f"  best rl_min_db={_fmt(run['best_rl_min_db'])} "
            f"fb_db={_fmt(run['best_fb_db'])} "
            f"fwd_eff_db={_fmt(run['best_fwd_eff_db'])} "
            f"eta={_fmt(run['best_eta_rad'])}"
        )
        print(
            f"  last rl_min_db={_fmt(last.get('rl_min_db'))} "
            f"eta={_fmt(last.get('eta_rad'))} "
            f"fwd_eff_db={_fmt(last.get('fwd_eff_db'))}"
        )
        cpu_dirs = opt_to_cpu.get(run["opt_run"], [])
        cpu_dir = _choose_latest_cpu_run(cpu_dirs)
        print(f"  cpu_eval_dir={cpu_dir}")
        if cpu_dir and cpu_dir in openems_map:
            openems_run, metrics = openems_map[cpu_dir]
            print(
                "  openems_run="
                f"{openems_run} rl_min={_fmt(metrics.get('rl_min_in_band_db'))} "
                f"gain={_fmt(metrics.get('gain_fwd_realized_db'))} "
                f"fb={_fmt(metrics.get('fb_db'))} "
                f"zin={_fmt(metrics.get('zin_f0_real_ohm'))}+j{_fmt(metrics.get('zin_f0_imag_ohm'))} "
                f"rad_eff={_fmt(metrics.get('rad_eff'))}"
            )
        else:
            print("  openems_run=not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

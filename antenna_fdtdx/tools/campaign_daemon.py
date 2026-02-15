"""Autopilot queue for optimize_topology runs with periodic CPU checks."""

from __future__ import annotations

import argparse
import math
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from tools import threshold_watch as tw


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, data: Dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _now() -> float:
    return time.time()


def _append_jsonl(path: str, payload: Dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_best_meta(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _float_or_none(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _swap_dir(tmp_dir: str, dest_dir: str, log: callable) -> bool:
    backup_dir = f"{dest_dir}.bak"
    if os.path.isdir(backup_dir):
        try:
            shutil.rmtree(backup_dir)
        except Exception as exc:
            log(f"best snapshot cleanup failed for {backup_dir}: {exc}")
            return False
    if os.path.isdir(dest_dir):
        try:
            os.rename(dest_dir, backup_dir)
        except Exception as exc:
            log(f"best snapshot move failed for {dest_dir}: {exc}")
            return False
    try:
        os.rename(tmp_dir, dest_dir)
    except Exception as exc:
        log(f"best snapshot swap failed for {dest_dir}: {exc}")
        if os.path.isdir(backup_dir):
            try:
                os.rename(backup_dir, dest_dir)
            except Exception as restore_exc:
                log(f"best snapshot restore failed for {dest_dir}: {restore_exc}")
        return False
    if os.path.isdir(backup_dir):
        try:
            shutil.rmtree(backup_dir)
        except Exception as exc:
            log(f"best snapshot cleanup failed for {backup_dir}: {exc}")
            return False
    return True


def _copy_run_files(src_dir: str, dest_dir: str, log: callable) -> bool:
    if not os.path.isdir(src_dir):
        log(f"best snapshot source missing: {src_dir}")
        return False
    tmp_dir = f"{dest_dir}.tmp"
    if os.path.isdir(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except Exception as exc:
            log(f"best snapshot temp cleanup failed for {tmp_dir}: {exc}")
            return False
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception as exc:
        log(f"best snapshot temp mkdir failed for {tmp_dir}: {exc}")
        return False
    ok = True
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(tmp_dir, name)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as exc:
            log(f"best snapshot copy failed for {src_path}: {exc}")
            ok = False
            break
    if not ok:
        try:
            shutil.rmtree(tmp_dir)
        except Exception as exc:
            log(f"best snapshot temp cleanup failed for {tmp_dir}: {exc}")
        return False
    return _swap_dir(tmp_dir, dest_dir, log)


def _default_port_reference(model_name: str) -> Optional[str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    calib_root = os.path.join(repo_root, "calibration")
    direct = os.path.join(calib_root, f"port_reference_{model_name}.json")
    if os.path.exists(direct):
        return direct
    if model_name.startswith("dir24"):
        fallback = os.path.join(calib_root, "port_reference_dir24_long.json")
        if os.path.exists(fallback):
            return fallback
    if model_name.startswith("dir5"):
        fallback = os.path.join(calib_root, "port_reference_dir5_fast.json")
        if os.path.exists(fallback):
            return fallback
    return None


def _port_reference_path(repo_root: str, model_name: str) -> str:
    return os.path.join(repo_root, "calibration", f"port_reference_{model_name}.json")


def _find_extra_arg(extra: List, flag: str) -> Optional[str]:
    for idx, val in enumerate(extra or []):
        if str(val) == flag and idx + 1 < len(extra):
            return str(extra[idx + 1])
    return None


def _resolve_rl_mode(run: Dict, defaults: Dict) -> str:
    rl_mode = run.get("rl_mode") or defaults.get("rl_mode")
    extra = run.get("extra_args") or defaults.get("extra_args") or []
    if rl_mode is None:
        rl_mode = _find_extra_arg(extra, "--rl-mode")
    return str(rl_mode) if rl_mode else "vi"


def _resolve_port_reference(run: Dict, defaults: Dict) -> Optional[str]:
    port_reference = run.get("port_reference") or defaults.get("port_reference")
    extra = run.get("extra_args") or defaults.get("extra_args") or []
    if port_reference is None:
        port_reference = _find_extra_arg(extra, "--port-reference")
    return port_reference


def _prune_port_reference_jobs(state: Dict, log: callable) -> set[int]:
    jobs = state.setdefault("port_reference_jobs", {})
    active_gpus: set[int] = set()
    for model_name, job in list(jobs.items()):
        pid = job.get("pid")
        gpu = job.get("gpu")
        out_path = job.get("out_path")
        if pid and _pid_alive(pid):
            if gpu is not None:
                active_gpus.add(int(gpu))
            continue
        jobs.pop(model_name, None)
        if out_path and os.path.exists(out_path):
            log(f"port_reference ready: {model_name} ({out_path})")
        else:
            log(f"port_reference failed: {model_name} (pid {pid})")
    return active_gpus


def _ensure_port_reference(
    repo_root: str,
    python_bin: str,
    model_name: str,
    gpu: Optional[int],
    state: Dict,
    log: callable,
) -> tuple[bool, bool]:
    out_path = _port_reference_path(repo_root, model_name)
    if os.path.exists(out_path):
        return True, False
    jobs = state.setdefault("port_reference_jobs", {})
    job = jobs.get(model_name)
    if job and job.get("pid") and _pid_alive(job.get("pid")):
        return False, False
    if job:
        jobs.pop(model_name, None)
    if gpu is None:
        return False, False
    cmd = [
        python_bin,
        "-m",
        "tools.port_reference",
        "--model",
        model_name,
        "--quality",
        "coarse",
        "--backend",
        "gpu",
        "--force",
        "--out",
        out_path,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    log_name = _sanitize_name(f"port_reference_{model_name}")
    log_path = os.path.join(repo_root, f"nohup_{log_name}.out")
    with open(log_path, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=handle,
            stderr=handle,
            env=env,
            start_new_session=True,
        )
    jobs[model_name] = {
        "pid": proc.pid,
        "gpu": int(gpu),
        "start_time": _now(),
        "out_path": out_path,
        "log_path": log_path,
    }
    log(f"port_reference start: {model_name} on GPU{gpu} pid={proc.pid}")
    return False, True


def _iter_threshold_metrics(root: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not os.path.isdir(root):
        return entries
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue
        if metrics.get("valid") is False:
            continue
        model = metrics.get("model")
        rl = metrics.get("rl_min_in_band_db")
        fb = metrics.get("fb_db")
        if model is None or rl is None or fb is None:
            continue
        entries.append(
            {
                "run_dir": run_dir,
                "run_id": name,
                "model": str(model),
                "rl": float(rl),
                "fb": float(fb),
            }
        )
    return entries


def _load_threshold_params(
    run_dir: str,
    np_mod,
    log,
    target_shape: Optional[Tuple[int, int, int]],
    allow_resample: bool,
    resample_method: str,
) -> Optional[object]:
    geom_path = os.path.join(run_dir, "geometry.npz")
    if not os.path.exists(geom_path):
        return None
    try:
        data = np_mod.load(geom_path)
    except Exception as exc:
        log(f"hybrid: failed to load {geom_path}: {exc}")
        return None
    if "params" in data:
        params = data["params"]
    elif "device_params" in data:
        params = data["device_params"]
    else:
        log(f"hybrid: no params in {geom_path}")
        return None
    params = params.astype(np_mod.float32)
    if target_shape and params.shape != target_shape:
        if not allow_resample:
            log(
                f"hybrid: shape mismatch {params.shape} vs {target_shape} for {run_dir}"
            )
            return None
        try:
            from sim.params import resample_params

            params = resample_params(params, target_shape, method=resample_method)
        except Exception as exc:
            log(f"hybrid: resample failed for {run_dir}: {exc}")
            return None
    return params


def _mix_params(
    params_rl,
    params_fb,
    np_mod,
    method: str,
    feed_frac: float,
    blend_alpha: float,
    binarize: bool,
    threshold: float,
):
    if params_rl is None or params_fb is None:
        return None
    if method == "split_feed":
        nx = params_rl.shape[0]
        cut = int(nx * feed_frac)
        cut = max(1, min(nx - 1, cut))
        out = params_fb.copy()
        out[:cut, :, :] = params_rl[:cut, :, :]
    elif method == "blend":
        out = blend_alpha * params_rl + (1.0 - blend_alpha) * params_fb
    elif method == "union":
        out = np_mod.maximum(params_rl, params_fb)
    elif method == "intersect":
        out = np_mod.minimum(params_rl, params_fb)
    else:
        return None
    out = np_mod.clip(out, 0.0, 1.0)
    if binarize:
        out = (out >= threshold).astype(np_mod.float32)
    return out


def _maybe_generate_hybrids(
    queue: Dict,
    state: Dict,
    queue_path: str,
    repo_root: str,
    log,
) -> int:
    cfg = queue.get("hybrid", {})
    if not cfg.get("enabled", False):
        return 0
    try:
        interval_done = int(cfg.get("interval_done", 0))
    except (TypeError, ValueError):
        interval_done = 0
    if interval_done <= 0:
        return 0
    done_count = sum(1 for v in state.get("runs", {}).values() if v.get("status") == "done")
    hybrid_state = state.setdefault("hybrid", {})
    last_done = int(hybrid_state.get("last_done", 0))
    if done_count - last_done < interval_done:
        return 0
    try:
        import numpy as np
    except Exception as exc:
        log(f"hybrid: numpy unavailable ({exc})")
        hybrid_state["last_done"] = done_count
        return 0

    source_root = str(cfg.get("source_root", os.path.join(repo_root, "runs_threshold")))
    entries = _iter_threshold_metrics(source_root)
    if not entries:
        hybrid_state["last_done"] = done_count
        return 0

    queue_models = {run.get("model") for run in queue.get("runs", []) if run.get("model")}
    if cfg.get("queue_models_only", True):
        entries = [entry for entry in entries if entry["model"] in queue_models]
    if not entries:
        hybrid_state["last_done"] = done_count
        return 0

    by_model: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        by_model.setdefault(entry["model"], []).append(entry)

    try:
        min_rl = float(cfg.get("min_rl_db")) if cfg.get("min_rl_db") is not None else None
    except (TypeError, ValueError):
        min_rl = None
    try:
        min_fb = float(cfg.get("min_fb_db")) if cfg.get("min_fb_db") is not None else None
    except (TypeError, ValueError):
        min_fb = None
    fallback_topk = bool(cfg.get("fallback_topk", True))
    max_per_cycle = int(cfg.get("max_per_cycle", 1))
    max_total = int(cfg.get("max_total", 0))
    seed_base = int(cfg.get("seed_base", 9000))
    feed_frac = float(cfg.get("feed_frac", 0.4))
    blend_alpha = float(cfg.get("blend_alpha", 0.6))
    binarize = bool(cfg.get("binarize", True))
    threshold = float(cfg.get("threshold", 0.5))
    allow_resample = bool(cfg.get("allow_resample", False))
    resample_method = str(cfg.get("resample_method", "nearest"))
    methods = cfg.get("methods") or ["split_feed"]
    if isinstance(methods, str):
        methods = [methods]

    existing_run_ids = {_run_id(run) for run in queue.get("runs", [])}
    generated = set(hybrid_state.get("generated", []))
    hybrid_count = int(hybrid_state.get("count", 0))
    existing_hybrids = sum(1 for run_id in existing_run_ids if str(run_id).startswith("hybrid_"))
    if existing_hybrids > hybrid_count:
        hybrid_count = existing_hybrids
    if max_total and hybrid_count >= max_total:
        hybrid_state["last_done"] = done_count
        return 0

    hybrid_root = str(cfg.get("hybrid_root", os.path.join(repo_root, "hybrid_seeds")))
    os.makedirs(hybrid_root, exist_ok=True)

    new_runs: List[Dict[str, object]] = []
    for model, entries in sorted(by_model.items()):
        entries_sorted_rl = sorted(entries, key=lambda e: float(e["rl"]), reverse=True)
        entries_sorted_fb = sorted(entries, key=lambda e: float(e["fb"]), reverse=True)
        rl_candidates = entries_sorted_rl
        fb_candidates = entries_sorted_fb
        if min_rl is not None:
            rl_candidates = [e for e in entries_sorted_rl if float(e["rl"]) >= min_rl]
        if min_fb is not None:
            fb_candidates = [e for e in entries_sorted_fb if float(e["fb"]) >= min_fb]
        if fallback_topk and not rl_candidates:
            rl_candidates = entries_sorted_rl
        if fallback_topk and not fb_candidates:
            fb_candidates = entries_sorted_fb
        if not rl_candidates or not fb_candidates:
            continue
        rl_entry = rl_candidates[0]
        fb_entry = fb_candidates[0]
        if rl_entry["run_id"] == fb_entry["run_id"]:
            continue
        params_rl = _load_threshold_params(
            rl_entry["run_dir"],
            np,
            log,
            None,
            allow_resample,
            resample_method,
        )
        if params_rl is None:
            continue
        params_fb = _load_threshold_params(
            fb_entry["run_dir"],
            np,
            log,
            params_rl.shape,
            allow_resample,
            resample_method,
        )
        if params_fb is None:
            continue
        for method in methods:
            if max_total and hybrid_count + len(new_runs) >= max_total:
                break
            if max_per_cycle and len(new_runs) >= max_per_cycle:
                break
            key = f"{model}:{rl_entry['run_id']}:{fb_entry['run_id']}:{method}"
            if key in generated:
                continue
            mixed = _mix_params(
                params_rl,
                params_fb,
                np,
                method,
                feed_frac,
                blend_alpha,
                binarize,
                threshold,
            )
            if mixed is None:
                continue
            seed = seed_base + hybrid_count + len(new_runs)
            name = f"hybrid_{model}_{method}_{rl_entry['run_id'][:6]}_{fb_entry['run_id'][:6]}_{seed}"
            if name in existing_run_ids:
                continue
            params_path = os.path.join(hybrid_root, f"{name}.npy")
            np.save(params_path, mixed.astype(np.float32))
            meta_path = os.path.join(hybrid_root, f"{name}.json")
            meta = {
                "model": model,
                "method": method,
                "seed": seed,
                "parent_rl": rl_entry["run_dir"],
                "parent_fb": fb_entry["run_dir"],
                "parent_rl_metric": float(rl_entry["rl"]),
                "parent_fb_metric": float(fb_entry["fb"]),
            }
            try:
                _save_json(meta_path, meta)
            except Exception:
                pass
            run = {
                "name": name,
                "model": model,
                "seed": seed,
                "init_params": params_path,
            }
            overrides = cfg.get("run_overrides", {})
            if isinstance(overrides, dict):
                run.update(overrides)
            new_runs.append(run)
            generated.add(key)
            existing_run_ids.add(name)
        if max_per_cycle and len(new_runs) >= max_per_cycle:
            break

    hybrid_state["last_done"] = done_count
    if not new_runs:
        return 0
    for run in new_runs:
        run_id = _run_id(run)
        state.setdefault("runs", {}).setdefault(
            run_id,
            {
                "status": "pending",
                "pid": None,
                "opt_run": None,
                "gpu": run.get("gpu"),
                "last_cpu_eval_epoch": -1,
                "last_cpu_eval_by_threshold": {},
            },
        )
    queue.setdefault("runs", []).extend(new_runs)
    hybrid_state["count"] = hybrid_count + len(new_runs)
    hybrid_state["generated"] = list(generated)
    _save_json(queue_path, queue)
    log(f"hybrid: queued {len(new_runs)} runs (count={hybrid_state['count']})")
    return len(new_runs)


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _score_penalty_raw_rl(rl_raw: Optional[float], target_db: float) -> float:
    if rl_raw is None or not isinstance(rl_raw, (int, float)):
        return target_db
    return max(0.0, target_db - float(rl_raw))


def _score_penalty_zin(zin_real: Optional[float], min_real: float) -> float:
    if zin_real is None or not isinstance(zin_real, (int, float)):
        return min_real
    return max(0.0, min_real - float(zin_real)) / max(min_real, 1e-6)


def _score_penalty_bw(bw_frac: Optional[float], target_frac: float) -> float:
    if bw_frac is None or not isinstance(bw_frac, (int, float)):
        return target_frac
    return max(0.0, target_frac - float(bw_frac))


def _should_update_best(meta: Optional[Dict], score: float) -> bool:
    best_score = _float_or_none(meta.get("score") if meta else None)
    if best_score is None:
        return True
    return score > best_score


def _update_best_snapshot(
    kind: str,
    run_id: str,
    opt_run: Optional[str],
    score: float,
    meta: Dict,
    source_dir: str,
    log: callable,
) -> bool:
    if not opt_run:
        log(f"{run_id} best {kind} skipped (missing opt_run)")
        return False
    if not math.isfinite(score):
        log(f"{run_id} best {kind} skipped (invalid score)")
        return False
    best_dir = os.path.join(opt_run, f"best_{kind}")
    best_meta_path = os.path.join(opt_run, f"best_{kind}.json")
    best_meta = _load_best_meta(best_meta_path)
    if not _should_update_best(best_meta, score):
        return False
    if not _copy_run_files(source_dir, best_dir, log):
        log(f"{run_id} best {kind} snapshot failed")
        return False
    best_payload = dict(meta)
    best_payload.update(
        {
            "score": score,
            "run_id": run_id,
            "source_dir": source_dir,
            "updated_ts": _now(),
        }
    )
    try:
        _save_json(best_meta_path, best_payload)
    except Exception as exc:
        log(f"{run_id} best {kind} meta write failed: {exc}")
        return False
    log(f"{run_id} best {kind} updated score={score:.2f}")
    return True


def _score_cpu_eval(entry: Dict, scoring: Dict) -> Tuple[float, Dict[str, float]]:
    use_match = bool(scoring.get("use_match_metrics", True))
    rl_raw = entry.get("rl_min_in_band_db")
    if rl_raw is None:
        rl_raw = entry.get("rl_min_db")
    if use_match:
        match_source, rl_match, bw_frac = _select_match_metrics(entry)
    else:
        match_source, rl_match, bw_frac = "raw", rl_raw, None
    fb = entry.get("fb_db") or 0.0
    if fb > 50.0:
        fb = 50.0
    elif fb < -50.0:
        fb = -50.0
    eta = entry.get("eta_rad") or 0.0
    if eta < 0.0:
        eta = 0.0
    elif eta > 1.0:
        eta = 1.0
    zin_real = entry.get("zin_f0_real_ohm")

    raw_target = float(scoring.get("raw_rl_penalty_db", 3.0))
    bw_target = float(scoring.get("bw_frac_target", 0.95))
    min_zin_real = float(scoring.get("min_zin_real", 5.0))

    pen_raw = _score_penalty_raw_rl(rl_raw, raw_target)
    pen_zin = _score_penalty_zin(zin_real, min_zin_real)
    pen_bw = 0.0 if not use_match else _score_penalty_bw(bw_frac, bw_target)

    rl_match_val = float(rl_match) if rl_match is not None else -100.0
    bw_val = float(bw_frac) if bw_frac is not None else 0.0
    if not use_match:
        bw_val = 0.0
    score = (
        2.0 * rl_match_val
        + 15.0 * bw_val
        + 0.5 * max(0.0, float(fb))
        + 10.0 * float(eta)
        - 2.0 * pen_raw
        - 5.0 * pen_zin
        - 10.0 * pen_bw
    )
    details = {
        "match_source": match_source,
        "rl_match_db": rl_match_val,
        "rl_raw_db": float(rl_raw) if rl_raw is not None else None,
        "bw_frac": bw_val,
        "fb_db": float(fb),
        "eta_rad": float(eta),
        "pen_raw": pen_raw,
        "pen_zin": pen_zin,
        "pen_bw": pen_bw,
        "use_match_metrics": use_match,
    }
    return score, details


def _select_match_metrics(metrics: Dict) -> Tuple[str, Optional[float], Optional[float]]:
    if metrics.get("match3_rl_min_in_band_db") is not None:
        return (
            "match3",
            metrics.get("match3_rl_min_in_band_db"),
            metrics.get("match3_bandwidth_frac"),
        )
    rl_matched = metrics.get("rl_min_in_band_matched_db")
    if rl_matched is None:
        rl_matched = metrics.get("rl_min_matched_db")
    if rl_matched is not None:
        return (
            "lmatch",
            rl_matched,
            metrics.get("match_bandwidth_frac"),
        )
    rl_raw = metrics.get("rl_min_in_band_db")
    if rl_raw is None:
        rl_raw = metrics.get("rl_min_db")
    return ("raw", rl_raw, None)


def _score_openems(metrics: Dict, scoring: Dict) -> Tuple[float, Dict[str, float]]:
    use_match = bool(scoring.get("use_match_metrics", True))
    if use_match:
        match_source, rl_match, bw_frac = _select_match_metrics(metrics)
    else:
        match_source, rl_match, bw_frac = "raw", None, None
    direction_mode = str(scoring.get("direction_mode", "forward")).strip().lower()
    use_peak_direction = direction_mode in {"peak", "any", "agnostic"}
    rl_raw = metrics.get("rl_min_in_band_db")
    if rl_raw is None:
        rl_raw = metrics.get("rl_min_db")
    gain_peak = metrics.get("plane_gain_peak_realized_db")
    gain_forward = metrics.get("plane_gain_forward_realized_db")
    gain = gain_peak if gain_peak is not None else metrics.get("gain_fwd_realized_db")
    if gain_forward is None:
        gain_forward = metrics.get("gain_fwd_realized_db")
    rad_eff_pct = metrics.get("rad_eff_pct") or 0.0
    if not isinstance(rad_eff_pct, (int, float)) or not math.isfinite(float(rad_eff_pct)):
        rad_eff_pct = 0.0
    fb_peak = metrics.get("plane_fb_peak_db")
    fb_forward = metrics.get("plane_fb_forward_db")
    fb = fb_peak if fb_peak is not None else metrics.get("fb_db") or 0.0
    if fb_forward is None:
        fb_forward = metrics.get("fb_db") or 0.0
    zin_real = metrics.get("zin_f0_real_ohm")
    p_acc = metrics.get("p_acc_w")
    rad_eff = metrics.get("rad_eff")
    phi_delta = metrics.get("plane_phi_delta_deg")
    valid_power = metrics.get("valid_power")

    raw_target = float(scoring.get("raw_rl_penalty_db", 3.0))
    bw_target = float(scoring.get("bw_frac_target", 0.95))
    min_zin_real = float(scoring.get("min_zin_real", 5.0))

    if p_acc is None or not isinstance(p_acc, (int, float)) or p_acc <= 0.0:
        details = {
            "match_source": match_source,
            "invalid_reason": "p_acc",
            "p_acc_w": p_acc,
        }
        return -1e9, details
    if valid_power is False:
        details = {
            "match_source": match_source,
            "invalid_reason": "valid_power",
            "valid_power": valid_power,
        }
        return -1e9, details
    if rad_eff is None or not isinstance(rad_eff, (int, float)) or not (0.0 <= float(rad_eff) <= 2.0):
        details = {
            "match_source": match_source,
            "invalid_reason": "rad_eff",
            "rad_eff": rad_eff,
        }
        return -1e9, details
    if rad_eff_pct > 100.0:
        rad_eff_pct = 100.0

    pen_raw = _score_penalty_raw_rl(rl_raw, raw_target)
    pen_zin = _score_penalty_zin(zin_real, min_zin_real)
    pen_bw = 0.0 if not use_match else _score_penalty_bw(bw_frac, bw_target)

    if not use_match:
        rl_match = rl_raw
    rl_match_val = float(rl_match) if rl_match is not None else -100.0
    bw_val = float(bw_frac) if bw_frac is not None else 0.0
    if not use_match:
        bw_val = 0.0
    gain_val = float(gain) if gain is not None else float("nan")
    if not math.isfinite(gain_val):
        gain_val = -200.0
    gain_forward_val = float(gain_forward) if gain_forward is not None else float("nan")
    if not math.isfinite(gain_forward_val):
        gain_forward_val = -200.0
    fb_val = float(fb) if fb is not None else float("nan")
    if not math.isfinite(fb_val):
        fb_val = 0.0
    fb_forward_val = float(fb_forward) if fb_forward is not None else float("nan")
    if not math.isfinite(fb_forward_val):
        fb_forward_val = 0.0
    phi_tol = float(scoring.get("phi_peak_tolerance_deg", 30.0))
    phi_penalty = float(scoring.get("phi_peak_penalty_weight", 5.0))
    phi_delta_val = None
    if isinstance(phi_delta, (int, float)):
        phi_delta_val = float(phi_delta)
    if phi_delta_val is None or not math.isfinite(phi_delta_val):
        phi_delta_val = 180.0
    if use_peak_direction:
        gain_forward_val = gain_val
        fb_forward_val = fb_val
        pen_phi = 0.0
        phi_penalty = 0.0
    else:
        pen_phi = max(0.0, (phi_delta_val - phi_tol) / max(phi_tol, 1e-6))
    score = (
        2.0 * rl_match_val
        + 1.5 * gain_val
        + 0.5 * gain_forward_val
        + 0.1 * float(rad_eff_pct)
        + 1.0 * max(0.0, fb_val)
        + 0.5 * max(0.0, fb_forward_val)
        + 10.0 * bw_val
        - 2.0 * pen_raw
        - 5.0 * pen_zin
        - 10.0 * pen_bw
        - phi_penalty * pen_phi
    )
    details = {
        "direction_mode": direction_mode,
        "match_source": match_source,
        "rl_match_db": rl_match_val,
        "rl_raw_db": float(rl_raw) if rl_raw is not None else None,
        "bw_frac": bw_val,
        "gain_realized_db": gain_val,
        "gain_forward_realized_db": gain_forward_val,
        "rad_eff_pct": float(rad_eff_pct),
        "fb_db": float(fb_val),
        "fb_forward_db": float(fb_forward_val),
        "pen_raw": pen_raw,
        "pen_zin": pen_zin,
        "pen_bw": pen_bw,
        "phi_delta_deg": phi_delta_val,
        "pen_phi": pen_phi,
        "use_match_metrics": use_match,
    }
    return score, details


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        stat_path = f"/proc/{pid}/stat"
        if os.path.exists(stat_path):
            with open(stat_path, "r", encoding="utf-8") as handle:
                fields = handle.read().split()
            if len(fields) > 2 and fields[2] == "Z":
                return False
        return True
    except OSError:
        return False


def _gpu_status() -> Dict[int, Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    status: Dict[int, Dict[str, float]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx = int(parts[0])
        util = float(parts[1])
        mem = float(parts[2])
        status[idx] = {"util": util, "mem": mem}
    return status


def _gpu_idle(gpu_idx: int, status: Dict[int, Dict[str, float]]) -> bool:
    info = status.get(gpu_idx)
    if info is None:
        return False
    return info["util"] < 5.0 and info["mem"] < 1000.0


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _log_cpu_eval_result(
    results_path: str,
    run_id: str,
    opt_run: Optional[str],
    epoch: int,
    eval_entry: Dict,
    cpu_run_root: str,
    out_dir: str,
    quality: str,
    rl_mode: str,
    scoring: Dict,
) -> Tuple[float, Dict[str, float]]:
    score, score_details = _score_cpu_eval(eval_entry, scoring)
    payload = {
        "source": "cpu_eval",
        "ts": _now(),
        "run_id": run_id,
        "opt_run": opt_run,
        "epoch": int(epoch),
        "threshold": eval_entry.get("threshold"),
        "quality": quality,
        "rl_mode": rl_mode,
        "cpu_run_dir": os.path.join(cpu_run_root, out_dir),
        "rl_min_db": eval_entry.get("rl_min_db"),
        "rl_min_matched_db": eval_entry.get("rl_min_matched_db"),
        "match3_rl_min_db": eval_entry.get("match3_rl_min_in_band_db"),
        "rl_peak_db": eval_entry.get("rl_peak_db"),
        "f_peak_hz": eval_entry.get("f_peak_hz"),
        "fb_db": eval_entry.get("fb_db"),
        "eta_rad": eval_entry.get("eta_rad"),
        "zin_f0_real_ohm": eval_entry.get("zin_f0_real_ohm"),
        "zin_f0_imag_ohm": eval_entry.get("zin_f0_imag_ohm"),
        "match_bandwidth_hz": eval_entry.get("match_bandwidth_hz"),
        "match_bandwidth_frac": eval_entry.get("match_bandwidth_frac"),
        "match3_bandwidth_hz": eval_entry.get("match3_bandwidth_hz"),
        "match3_bandwidth_frac": eval_entry.get("match3_bandwidth_frac"),
        "fill_frac": eval_entry.get("fill_frac"),
        "score": score,
        "score_details": score_details,
    }
    _append_jsonl(results_path, payload)
    return score, score_details


def _early_stop_rule_met(
    rule: Dict,
    eval_entry: Dict,
    epoch: int,
    default_threshold: float,
) -> tuple[bool, str]:
    try:
        min_epoch = int(rule.get("min_epoch", 0) or 0)
    except (TypeError, ValueError):
        min_epoch = 0
    try:
        max_epoch = int(rule.get("max_epoch", 0) or 0)
    except (TypeError, ValueError):
        max_epoch = 0
    if min_epoch and (epoch + 1) < min_epoch:
        return False, ""
    if max_epoch and (epoch + 1) > max_epoch:
        return False, ""
    rule_thr = rule.get("threshold")
    eval_thr = eval_entry.get("threshold", default_threshold)
    if rule_thr is not None:
        try:
            rule_thr = float(rule_thr)
        except (TypeError, ValueError):
            return False, ""
        if eval_thr is None or abs(float(eval_thr) - rule_thr) > 1e-6:
            return False, ""
    rl_key = rule.get("rl_key", "rl_min_db")
    rl_val = eval_entry.get(rl_key)
    if rl_val is None:
        return False, ""
    min_rl = rule.get("rl_min_db")
    if min_rl is None:
        return False, ""
    try:
        min_rl = float(min_rl)
    except (TypeError, ValueError):
        return False, ""
    if float(rl_val) >= min_rl:
        return False, ""
    reason = f"{rl_key}={rl_val:.2f} < {min_rl:.2f}"
    if rule_thr is not None:
        reason += f" thr={float(rule_thr):.2f}"
    return True, reason


def _log_openems_results(
    results_path: str,
    openems_root: str,
    logged: set[str],
    scoring: Dict,
    cpu_run_index: Optional[Dict[str, Dict]],
    log: Optional[callable],
) -> List[str]:
    newly_logged: List[str] = []
    if not os.path.isdir(openems_root):
        return newly_logged
    for name in os.listdir(openems_root):
        run_dir = os.path.join(openems_root, name)
        if run_dir in logged or not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue
        meta_path = os.path.join(run_dir, "fdtdx_meta.json")
        fdtdx_run = None
        model = None
        if os.path.exists(meta_path):
            try:
                meta = _load_json(meta_path)
                fdtdx_run = meta.get("fdtdx_run")
                model = meta.get("model")
            except Exception:
                fdtdx_run = None
                model = None
        if model is None and fdtdx_run:
            params_path = os.path.join(fdtdx_run, "params.json")
            if os.path.exists(params_path):
                try:
                    fdtdx_params = _load_json(params_path)
                    model = fdtdx_params.get("model")
                except Exception:
                    model = None
        score, score_details = _score_openems(metrics, scoring)
        payload = {
            "source": "openems",
            "ts": _now(),
            "openems_run": run_dir,
            "fdtdx_run": fdtdx_run,
            "model": model,
            "rl_min_db": metrics.get("rl_min_in_band_db"),
            "rl_min_matched_db": metrics.get("rl_min_in_band_matched_db"),
            "match3_rl_min_db": metrics.get("match3_rl_min_in_band_db"),
            "rl_peak_db": metrics.get("rl_peak_in_band_db"),
            "fb_db": metrics.get("fb_db"),
            "gain_fwd_db": metrics.get("gain_fwd_db"),
            "gain_fwd_realized_db": metrics.get("gain_fwd_realized_db"),
            "p_acc_w": metrics.get("p_acc_w"),
            "rad_eff": metrics.get("rad_eff"),
            "rad_eff_pct": metrics.get("rad_eff_pct"),
            "effective_aperture_m2": metrics.get("effective_aperture_m2"),
            "zin_f0_real_ohm": metrics.get("zin_f0_real_ohm"),
            "zin_f0_imag_ohm": metrics.get("zin_f0_imag_ohm"),
            "f0_hz": metrics.get("f0_hz"),
            "band_hz": metrics.get("band_hz"),
            "match_bandwidth_frac": metrics.get("match_bandwidth_frac"),
            "match3_bandwidth_frac": metrics.get("match3_bandwidth_frac"),
            "score": score,
            "score_details": score_details,
        }
        _append_jsonl(results_path, payload)
        logged.add(run_dir)
        newly_logged.append(run_dir)
        if fdtdx_run and cpu_run_index:
            run_key = os.path.abspath(fdtdx_run)
            mapped = cpu_run_index.get(run_key)
            if mapped:
                opt_run = mapped.get("opt_run")
                run_id = mapped.get("run_id")
                if run_id and log:
                    if score_details.get("invalid_reason"):
                        log(
                            f"{run_id} best openems skipped (invalid {score_details.get('invalid_reason')})"
                        )
                    else:
                        _update_best_snapshot(
                            "openems",
                            str(run_id),
                            opt_run,
                            float(score),
                            {
                                "fdtdx_run": fdtdx_run,
                                "openems_run": run_dir,
                                "score_details": score_details,
                            },
                            run_dir,
                            log,
                        )
            elif log:
                log(f"openems best skipped (unmapped fdtdx_run {fdtdx_run})")
    return newly_logged


def _run_id(run: Dict) -> str:
    name = run.get("name")
    if name:
        return str(name)
    model = run.get("model", "unknown")
    seed = run.get("seed", 0)
    return f"{model}_seed{seed}"


def _build_command(
    run: Dict,
    defaults: Dict,
    python_bin: str,
    run_root: Optional[str],
) -> List[str]:
    def _get(key: str, default=None):
        return run.get(key, defaults.get(key, default))

    cmd = [
        python_bin,
        "-m",
        "opt.optimize_topology",
        "--model",
        str(_get("model")),
        "--quality",
        str(_get("quality", "coarse")),
        "--backend",
        str(_get("backend", "gpu")),
        "--grad-method",
        str(_get("grad_method", "checkpointed")),
        "--grad-checkpoints",
        str(int(_get("grad_checkpoints", 8))),
        "--schedule",
        str(_get("schedule", "overnight_binary")),
        "--seed",
        str(_get("seed", 0)),
    ]
    if _get("grad_stall_epochs") is not None:
        cmd.extend(["--grad-stall-epochs", str(int(_get("grad_stall_epochs")))])
    if _get("grad_stall_threshold") is not None:
        cmd.extend(["--grad-stall-threshold", str(float(_get("grad_stall_threshold")))])
    if _get("nan_grad_mode"):
        cmd.extend(["--nan-grad-mode", str(_get("nan_grad_mode"))])
    if _get("nan_grad_epochs") is not None:
        cmd.extend(["--nan-grad-epochs", str(int(_get("nan_grad_epochs")))])
    if _get("nan_grad_threshold") is not None:
        cmd.extend(["--nan-grad-threshold", str(float(_get("nan_grad_threshold")))])
    if _get("device_non_pec"):
        cmd.append("--device-non-pec")
    if _get("device_non_pec") and _get("device_eps") is not None:
        cmd.extend(["--device-eps", str(float(_get("device_eps")))])
    if _get("search_mode"):
        cmd.extend(["--search-mode", str(_get("search_mode"))])
    if _get("random_scale") is not None:
        cmd.extend(["--random-scale", str(float(_get("random_scale")))])
    if _get("random_decay") is not None:
        cmd.extend(["--random-decay", str(float(_get("random_decay")))])
    if _get("random_binary"):
        cmd.append("--random-binary")
    if _get("random_threshold") is not None:
        cmd.extend(["--random-threshold", str(float(_get("random_threshold")))])
    if _get("random_accept"):
        cmd.extend(["--random-accept", str(_get("random_accept"))])
    if _get("epsilon_random_prob") is not None:
        cmd.extend(["--epsilon-random-prob", str(float(_get("epsilon_random_prob")))])
    if _get("epsilon_random_scale") is not None:
        cmd.extend(["--epsilon-random-scale", str(float(_get("epsilon_random_scale")))])
    if _get("grad_noise_prob") is not None:
        cmd.extend(["--grad-noise-prob", str(float(_get("grad_noise_prob")))])
    if _get("grad_noise_scale") is not None:
        cmd.extend(["--grad-noise-scale", str(float(_get("grad_noise_scale")))])
    if _get("band_sample_prob") is not None:
        cmd.extend(["--band-sample-prob", str(float(_get("band_sample_prob")))])
    if _get("design_dropout_prob") is not None:
        cmd.extend(["--design-dropout-prob", str(float(_get("design_dropout_prob")))])
    if _get("w_raw_rl_pen") is not None:
        cmd.extend(["--w-raw-rl-pen", str(float(_get("w_raw_rl_pen")))])
    if _get("raw_rl_floor_db") is not None:
        cmd.extend(["--raw-rl-floor-db", str(float(_get("raw_rl_floor_db")))])
    if _get("schedule_json"):
        cmd.extend(["--schedule-json", str(_get("schedule_json"))])
    if _get("init_params"):
        cmd.extend(["--init-params", str(_get("init_params"))])
    if _get("init_mode"):
        cmd.extend(["--init-mode", str(_get("init_mode"))])
    if _get("init_blob_density") is not None:
        cmd.extend(["--init-blob-density", str(float(_get("init_blob_density")))])
    if _get("init_blob_smooth_iters") is not None:
        cmd.extend(["--init-blob-smooth-iters", str(int(_get("init_blob_smooth_iters")))])
    if _get("init_walk_length_frac") is not None:
        cmd.extend(["--init-walk-length-frac", str(float(_get("init_walk_length_frac")))])
    if _get("init_walk_length_cells") is not None:
        cmd.extend(["--init-walk-length-cells", str(int(_get("init_walk_length_cells")))])
    if _get("init_walk_branches") is not None:
        cmd.extend(["--init-walk-branches", str(int(_get("init_walk_branches")))])
    if _get("init_walk_thickness") is not None:
        cmd.extend(["--init-walk-thickness", str(int(_get("init_walk_thickness")))])
    if _get("init_walk_turn_prob") is not None:
        cmd.extend(["--init-walk-turn-prob", str(float(_get("init_walk_turn_prob")))])
    if _get("init_grid_size") is not None:
        cmd.extend(["--init-grid-size", str(int(_get("init_grid_size")))])
    if _get("init_grid_connect_prob") is not None:
        cmd.extend(["--init-grid-connect-prob", str(float(_get("init_grid_connect_prob")))])
    if _get("init_grid_thickness") is not None:
        cmd.extend(["--init-grid-thickness", str(int(_get("init_grid_thickness")))])
    if _get("threshold_every", 0):
        cmd.extend(["--threshold-every", str(int(_get("threshold_every")))])
    if _get("thresholds"):
        cmd.extend(["--thresholds", str(_get("thresholds"))])
    if _get("threshold_beta"):
        cmd.extend(["--threshold-beta", str(float(_get("threshold_beta")))])
    if _get("threshold_stop_below") is not None:
        cmd.extend(["--threshold-stop-below", str(float(_get("threshold_stop_below")))])
    if _get("threshold_stop_after"):
        cmd.extend(["--threshold-stop-after", str(int(_get("threshold_stop_after")))])
    if _get("freeze_x_frac"):
        cmd.extend(["--freeze-x-frac", str(float(_get("freeze_x_frac")))])
    if _get("freeze_x_cells"):
        cmd.extend(["--freeze-x-cells", str(int(_get("freeze_x_cells")))])
    if _get("save_every"):
        cmd.extend(["--save-every", str(int(_get("save_every")))])
    if _get("final_eval"):
        cmd.append("--final-eval")
    if _get("port_calibration"):
        cmd.extend(["--port-calibration", str(_get("port_calibration"))])
    if _get("port_calibration_metric"):
        cmd.extend(["--port-calibration-metric", str(_get("port_calibration_metric"))])
    port_reference = _get("port_reference")
    if port_reference:
        cmd.extend(["--port-reference", str(port_reference)])
    if run_root:
        cmd.extend(["--run-root", run_root])
    extra = _get("extra_args", [])
    if extra:
        cmd.extend([str(x) for x in extra])
    if not port_reference:
        rl_mode = _get("rl_mode")
        if rl_mode is None:
            rl_mode = _find_extra_arg(extra, "--rl-mode")
        if rl_mode == "thevenin":
            inferred = _default_port_reference(str(_get("model")))
            if inferred:
                cmd.extend(["--port-reference", inferred])
    return cmd


def _find_opt_run(
    opt_root: str,
    model: str,
    seed: int,
    started_after: float,
    init_mode: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Optional[str]:
    if not os.path.isdir(opt_root):
        return None
    candidates = []
    for run_id in os.listdir(opt_root):
        cfg_path = os.path.join(opt_root, run_id, "config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            mtime = os.path.getmtime(cfg_path)
        except OSError:
            continue
        if mtime < started_after - 300:
            continue
        try:
            cfg = _load_json(cfg_path)
        except Exception:
            continue
        if cfg.get("model") != model:
            continue
        if int(cfg.get("seed", -1)) != seed:
            continue
        if init_mode and cfg.get("init_mode") != init_mode:
            continue
        if run_tag and cfg.get("run_tag") != run_tag:
            continue
        candidates.append((mtime, run_id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(opt_root, candidates[0][1])


def _launch_openems(
    python_bin: str,
    fdtdx_run: str,
    openems_cfg: Dict,
    log_path: str,
) -> Optional[int]:
    if not openems_cfg.get("enabled", False):
        return None
    cmd = [
        python_bin,
        "-m",
        "tools.run_openems_from_fdtdx",
        "--fdtdx-run",
        fdtdx_run,
        "--quality",
        str(openems_cfg.get("quality", "fast")),
        "--port-mode",
        str(openems_cfg.get("port_mode", "fdtdx")),
        "--port-type",
        str(openems_cfg.get("port_type", "msl")),
        "--metrics",
    ]
    if openems_cfg.get("threshold") is not None:
        cmd.extend(["--threshold", str(float(openems_cfg["threshold"]))])
    if openems_cfg.get("mirror_fdtdx", True):
        cmd.append("--mirror-fdtdx")
    if openems_cfg.get("match_fdtdx_mesh", True):
        cmd.append("--match-fdtdx-mesh")
    if openems_cfg.get("no_clip_to_wedge", True):
        cmd.append("--no-clip-to-wedge")
    if openems_cfg.get("max_cell_mm") is not None:
        cmd.extend(["--max-cell-mm", str(float(openems_cfg["max_cell_mm"]))])
    if openems_cfg.get("snap_mm") is not None:
        cmd.extend(["--snap-mm", str(float(openems_cfg["snap_mm"]))])
    if openems_cfg.get("smooth_ratio") is not None:
        cmd.extend(["--smooth-ratio", str(float(openems_cfg["smooth_ratio"]))])
    if openems_cfg.get("nr_ts") is not None:
        cmd.extend(["--nr-ts", str(int(openems_cfg["nr_ts"]))])
    if openems_cfg.get("end_criteria") is not None:
        cmd.extend(["--end-criteria", str(float(openems_cfg["end_criteria"]))])
    if openems_cfg.get("force", False):
        cmd.append("--force")
    if openems_cfg.get("no_vtk", False):
        cmd.append("--no-vtk")
    elif openems_cfg.get("vtk", False):
        cmd.append("--vtk")
    if openems_cfg.get("prune", False):
        cmd.append("--prune")
    overlap = float(openems_cfg.get("overlap_mm", 0.5))
    if overlap > 0:
        cmd.extend(["--overlap-mm", str(overlap)])
    if openems_cfg.get("match3", False):
        cmd.append("--match3")
        cmd.extend(["--match3-l-min-nh", str(float(openems_cfg.get("match3_l_min_nh", 0.5)))])
        cmd.extend(["--match3-l-max-nh", str(float(openems_cfg.get("match3_l_max_nh", 20.0)))])
        cmd.extend(["--match3-c-min-pf", str(float(openems_cfg.get("match3_c_min_pf", 0.2)))])
        cmd.extend(["--match3-c-max-pf", str(float(openems_cfg.get("match3_c_max_pf", 10.0)))])
        cmd.extend(["--match3-samples", str(int(openems_cfg.get("match3_samples", 8)))])

    env = dict(os.environ)
    openems_root_env = env.get("OPENEMS_ROOT")
    if openems_root_env:
        openems_lib_dir = os.path.join(openems_root_env, "lib")
        env["LD_LIBRARY_PATH"] = openems_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    with open(log_path, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.join(os.path.dirname(__file__), "..", "..", "antenna_opt"),
            stdout=handle,
            stderr=handle,
            env=env,
            start_new_session=True,
        )
    return proc.pid


def _export_fdtdx_geometry(
    python_bin: str,
    run_dir: str,
    threshold: float,
    log: callable,
) -> None:
    params_path = os.path.join(run_dir, "params.json")
    geom_path = os.path.join(run_dir, "geometry.npz")
    if not os.path.exists(params_path) or not os.path.exists(geom_path):
        log(f"export_geometry skipped: missing params/geometry in {run_dir}")
        return
    tag = f"thr{threshold:.2f}".replace(".", "p")
    out_path = os.path.join(run_dir, f"top_copper_{tag}.svg")
    cmd = [
        python_bin,
        "-m",
        "tools.export_geometry",
        "--run",
        run_dir,
        "--threshold",
        str(threshold),
        "--include-feed",
        "--out",
        out_path,
    ]
    proc = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), ".."), capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"export_geometry failed: {proc.stderr.strip()}")


def _collect_fdtdx_runs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    runs = []
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.exists(os.path.join(run_dir, "params.json")):
            continue
        if not os.path.exists(os.path.join(run_dir, "geometry.npz")):
            continue
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        runs.append(run_dir)
    runs.sort(key=lambda p: os.path.getmtime(os.path.join(p, "metrics.json")))
    return runs


def _index_openems_runs(openems_root: str) -> Dict[str, Dict[str, str]]:
    done: Dict[str, str] = {}
    pending: Dict[str, str] = {}
    if not os.path.isdir(openems_root):
        return {"done": done, "pending": pending}
    for name in os.listdir(openems_root):
        run_dir = os.path.join(openems_root, name)
        meta_path = os.path.join(run_dir, "fdtdx_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = _load_json(meta_path)
        except Exception:
            continue
        fdtdx_run = meta.get("fdtdx_run")
        if not fdtdx_run:
            continue
        fdtdx_run = os.path.abspath(fdtdx_run)
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            done[fdtdx_run] = run_dir
        else:
            pending[fdtdx_run] = run_dir
    return {"done": done, "pending": pending}


def _prune_openems_pids(state: Dict[str, Dict]) -> List[int]:
    active: List[int] = []
    for entry in state.get("runs", {}).values():
        pids = entry.get("openems_pids", [])
        alive = []
        for pid in pids:
            if _pid_alive(pid):
                alive.append(pid)
                active.append(pid)
        if alive != pids:
            entry["openems_pids"] = alive
    scan_jobs = state.setdefault("openems_scan_jobs", {})
    for job in scan_jobs.values():
        pid = job.get("pid")
        if pid and _pid_alive(pid):
            active.append(pid)
        elif job.get("status") == "running":
            job["status"] = "done"
    return active


def _run_archive_passes(
    python_bin: str,
    repo_root: str,
    openems_root: str,
    out_root: str,
    archive_cfg: Dict,
    log: callable,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "tools.archive_passes",
        "--openems-root",
        openems_root,
        "--out-root",
        out_root,
    ]
    if archive_cfg.get("min_rl_db") is not None:
        cmd.extend(["--min-rl-db", str(float(archive_cfg["min_rl_db"]))])
    if archive_cfg.get("min_fb_db") is not None:
        cmd.extend(["--min-fb-db", str(float(archive_cfg["min_fb_db"]))])
    if archive_cfg.get("min_gain_db") is not None:
        cmd.extend(["--min-gain-db", str(float(archive_cfg["min_gain_db"]))])
    if archive_cfg.get("min_zin_real") is not None:
        cmd.extend(["--min-zin-real", str(float(archive_cfg["min_zin_real"]))])
    if archive_cfg.get("max_zin_real") is not None:
        cmd.extend(["--max-zin-real", str(float(archive_cfg["max_zin_real"]))])
    if archive_cfg.get("max_zin_imag") is not None:
        cmd.extend(["--max-zin-imag", str(float(archive_cfg["max_zin_imag"]))])
    if archive_cfg.get("min_pacc_w") is not None:
        cmd.extend(["--min-pacc-w", str(float(archive_cfg["min_pacc_w"]))])
    if archive_cfg.get("min_rad_eff") is not None:
        cmd.extend(["--min-rad-eff", str(float(archive_cfg["min_rad_eff"]))])
    if archive_cfg.get("max_rad_eff") is not None:
        cmd.extend(["--max-rad-eff", str(float(archive_cfg["max_rad_eff"]))])
    if archive_cfg.get("use_matched_rl", False):
        cmd.append("--use-matched-rl")
    if archive_cfg.get("use_match3", False):
        cmd.append("--use-match3")
    if archive_cfg.get("min_bw_frac") is not None:
        cmd.extend(["--min-bw-frac", str(float(archive_cfg["min_bw_frac"]))])
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"archive_passes failed: {proc.stderr.strip()}")


def _cpu_eval_ready(
    opt_run: str,
    interval: int,
    last_epoch: int,
) -> Optional[int]:
    hist = os.path.join(opt_run, "history.jsonl")
    epoch = tw._read_last_epoch(hist)
    if epoch is None:
        return None
    if epoch <= last_epoch:
        return None
    if interval <= 0:
        return None
    # If we missed an interval, return the latest eligible epoch.
    target = ((epoch + 1) // interval) * interval - 1
    if target < 0:
        return None
    if target <= last_epoch:
        return None
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, help="Path to campaign_queue.json")
    parser.add_argument("--state", default=None, help="State JSON path")
    parser.add_argument("--log", default=None, help="Daemon log path")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_bin = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable
    opt_root = os.path.join(repo_root, "opt_runs")

    state_path = args.state or os.path.join(repo_root, "campaign_state.json")
    log_path = args.log or os.path.join(repo_root, "campaign_daemon.log")
    run_root_default = os.path.join(repo_root, "opt_runs")

    state: Dict[str, Dict] = {"runs": {}, "last_update": 0.0}
    if os.path.exists(state_path):
        try:
            state = _load_json(state_path)
        except Exception:
            state = {"runs": {}, "last_update": 0.0}

    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line)

    log("campaign_daemon start")

    cpu_pool: Optional[ThreadPoolExecutor] = None
    cpu_jobs: Dict[object, Dict[str, object]] = {}
    cpu_inflight_runs: set = set()

    while True:
        queue = _load_json(args.queue)
        defaults = queue.get("defaults", {})
        runs = queue.get("runs", [])
        openems_cfg = queue.get("openems", {})
        results_log = str(queue.get("results_log", os.path.join(repo_root, "campaign_results.jsonl")))
        if not os.path.isabs(results_log):
            results_log = os.path.abspath(os.path.join(repo_root, results_log))
        scoring_cfg = queue.get("scoring", {})
        cpu_cfg = queue.get("cpu_threshold", {})
        early_stop_rules = cpu_cfg.get("early_stop", [])
        if not isinstance(early_stop_rules, list):
            early_stop_rules = []
        cpu_interval = int(cpu_cfg.get("interval", 0))
        cpu_threshold = float(cpu_cfg.get("threshold", 0.5))
        raw_thresholds = cpu_cfg.get("thresholds")
        if raw_thresholds is None:
            cpu_thresholds = [cpu_threshold]
        elif isinstance(raw_thresholds, str):
            cpu_thresholds = [float(x) for x in raw_thresholds.split(",") if x.strip()]
        elif isinstance(raw_thresholds, (list, tuple)):
            cpu_thresholds = [float(x) for x in raw_thresholds]
        else:
            cpu_thresholds = [cpu_threshold]
        if not cpu_thresholds:
            cpu_thresholds = [cpu_threshold]
        if all(abs(th - cpu_threshold) > 1e-6 for th in cpu_thresholds):
            cpu_thresholds.append(cpu_threshold)
        cpu_quality = str(cpu_cfg.get("quality", "coarse"))
        cpu_rl_mode = str(cpu_cfg.get("rl_mode", "flux"))
        cpu_port_calibration = cpu_cfg.get("port_calibration") or defaults.get("port_calibration")
        cpu_calibration_metric = (
            cpu_cfg.get("port_calibration_metric")
            or defaults.get("port_calibration_metric")
            or "auto"
        )
        cpu_run_root = str(cpu_cfg.get("run_root", os.path.join(repo_root, "runs_threshold")))
        if not os.path.isabs(cpu_run_root):
            cpu_run_root = os.path.abspath(os.path.join(repo_root, cpu_run_root))
        cpu_params_resample = bool(cpu_cfg.get("params_resample", False))
        cpu_params_resample_method = str(cpu_cfg.get("params_resample_method", "nearest"))
        cpu_max_workers = int(cpu_cfg.get("max_workers", min(8, os.cpu_count() or 4)))
        cpu_max_inflight = int(cpu_cfg.get("max_inflight", cpu_max_workers))
        cpu_eval_on_done = bool(cpu_cfg.get("eval_on_done", True))
        cpu_stop_below = cpu_cfg.get("stop_below")
        cpu_stop_after = int(cpu_cfg.get("stop_after", 0))
        cpu_stale_limit = cpu_cfg.get("stale_limit")
        cpu_stop_density_min = cpu_cfg.get("stop_density_min")
        cpu_stop_density_max = cpu_cfg.get("stop_density_max")
        cpu_stop_density_after = int(cpu_cfg.get("stop_density_after", 0))
        os.makedirs(cpu_run_root, exist_ok=True)
        openems_root = str(
            openems_cfg.get(
                "openems_root",
                os.path.abspath(os.path.join(repo_root, "..", "antenna_opt", "runs_fdtdx")),
            )
        )
        if not os.path.isabs(openems_root):
            openems_root = os.path.abspath(os.path.join(repo_root, openems_root))
        openems_threshold = openems_cfg.get("threshold")
        if openems_threshold is None:
            openems_threshold = cpu_threshold
        openems_cfg = dict(openems_cfg)
        openems_cfg["threshold"] = openems_threshold
        scan_runs = bool(openems_cfg.get("scan_runs", False))
        scan_interval = float(openems_cfg.get("scan_interval", 600.0))
        openems_log_interval = float(openems_cfg.get("log_interval", 300.0))
        scan_max_per_loop = int(openems_cfg.get("scan_max_per_loop", 1))
        scan_roots = openems_cfg.get("scan_roots")
        if not scan_roots:
            scan_roots = [os.path.join(repo_root, "runs")]
        scan_roots = [
            path if os.path.isabs(path) else os.path.abspath(os.path.join(repo_root, path))
            for path in scan_roots
        ]
        archive_cfg = openems_cfg.get("archive", {})
        archive_enabled = bool(openems_cfg.get("auto_archive", False))
        archive_interval = float(openems_cfg.get("archive_interval", 1800.0))
        archive_out_root = str(openems_cfg.get("archive_root", os.path.join(repo_root, "passing_designs")))

        if cpu_interval > 0 and cpu_max_workers > 0 and cpu_pool is None:
            cpu_pool = ThreadPoolExecutor(max_workers=cpu_max_workers)

        for run in runs:
            run_id = _run_id(run)
            state["runs"].setdefault(
                run_id,
                {
                    "status": "pending",
                    "pid": None,
                    "opt_run": None,
                    "gpu": run.get("gpu"),
                    "last_cpu_eval_epoch": -1,
                    "last_cpu_eval_by_threshold": {},
                },
            )

        active_openems = _prune_openems_pids(state)
        max_openems = int(openems_cfg.get("max_concurrent", 1))

        gpu_status = _gpu_status()
        busy_gpus = set()
        for entry in state["runs"].values():
            if entry.get("status") != "running":
                continue
            pid = entry.get("pid")
            if not _pid_alive(pid):
                entry["status"] = "done"
                log(f"pid {pid} stale; marked done")
                continue
            if entry.get("gpu") is not None:
                busy_gpus.add(int(entry.get("gpu")))

        busy_gpus.update(_prune_port_reference_jobs(state, log))

        cpu_logged = set(state.setdefault("cpu_logged", []))
        if cpu_jobs:
            done_jobs = [future for future in cpu_jobs if future.done()]
            for future in done_jobs:
                meta = cpu_jobs.pop(future)
                run_id = str(meta["run_id"])
                epoch = int(meta["epoch"])
                thr_val = float(meta.get("threshold", cpu_threshold))
                thr_key = meta.get("thr_key", _threshold_key(thr_val))
                cpu_inflight_runs.discard((run_id, thr_key))
                entry = state["runs"].get(run_id)
                if entry is None:
                    continue
                try:
                    eval_entry, out_dir = future.result()
                except Exception as exc:
                    log(f"{run_id} cpu eval failed: {exc}")
                    continue
                log_key = f"{run_id}:{epoch}:{eval_entry.get('threshold')}"
                if log_key not in cpu_logged:
                    score, score_details = _log_cpu_eval_result(
                        results_log,
                        run_id,
                        entry.get("opt_run"),
                        epoch,
                        eval_entry,
                        cpu_run_root,
                        out_dir,
                        cpu_quality,
                        cpu_rl_mode,
                        scoring_cfg,
                    )
                    cpu_logged.add(log_key)
                    state["cpu_logged"] = list(cpu_logged)
                    cpu_run_dir = os.path.abspath(os.path.join(cpu_run_root, out_dir))
                    cpu_index = state.setdefault("cpu_run_index", {})
                    cpu_index[cpu_run_dir] = {
                        "run_id": run_id,
                        "opt_run": entry.get("opt_run"),
                        "epoch": int(epoch),
                        "threshold": float(eval_entry.get("threshold", thr_val)),
                    }
                    _update_best_snapshot(
                        "cpu_eval",
                        run_id,
                        entry.get("opt_run"),
                        float(score),
                        {
                            "epoch": int(epoch) + 1,
                            "threshold": float(eval_entry.get("threshold", thr_val)),
                            "score_details": score_details,
                        },
                        cpu_run_dir,
                        log,
                    )
                last_by_thr = entry.setdefault("last_cpu_eval_by_threshold", {})
                last_by_thr[thr_key] = max(int(last_by_thr.get(thr_key, -1)), epoch)
                is_default_thr = abs(float(eval_entry.get("threshold", thr_val)) - cpu_threshold) <= 1e-6
                if is_default_thr:
                    entry["last_cpu_eval_epoch"] = max(entry.get("last_cpu_eval_epoch", -1), epoch)
                early_stop_reason = None
                if early_stop_rules and not entry.get("early_stop"):
                    for rule in early_stop_rules:
                        triggered, reason = _early_stop_rule_met(
                            rule,
                            eval_entry,
                            epoch,
                            cpu_threshold,
                        )
                        if triggered:
                            early_stop_reason = reason
                            break
                    if early_stop_reason:
                        pid = entry.get("pid")
                        if pid and _pid_alive(pid):
                            try:
                                os.kill(pid, 15)
                            except OSError as exc:
                                log(f"{run_id} cpu early-stop failed to kill pid {pid}: {exc}")
                        entry["early_stop"] = {
                            "epoch": epoch + 1,
                            "threshold": eval_entry.get("threshold", thr_val),
                            "reason": early_stop_reason,
                        }
                        log(f"{run_id} cpu early-stop {early_stop_reason} epoch={epoch + 1}")
                early_stop_active = entry.get("early_stop") is not None
                if openems_cfg.get("enabled", False) and not early_stop_active:
                    allow_invalid = bool(openems_cfg.get("allow_invalid_cpu_eval", True))
                    hard_invalid_reasons = set(openems_cfg.get("hard_invalid_reasons") or [])
                    min_rl = float(openems_cfg.get("min_rl_db", 10.0))
                    use_match3 = bool(openems_cfg.get("use_match3", False))
                    if use_match3:
                        rl_key = "match3_rl_min_in_band_db"
                        bw_key = "match3_bandwidth_frac"
                    else:
                        rl_key = "rl_min_matched_db" if openems_cfg.get("use_matched_rl", False) else "rl_min_db"
                        bw_key = "match_bandwidth_frac"
                    rl_val = eval_entry.get(rl_key)
                    if rl_val is None and rl_key != "rl_min_db":
                        rl_val = eval_entry.get("rl_min_db")
                    fill_frac = eval_entry.get("fill_frac")
                    min_fill = openems_cfg.get("min_fill_frac")
                    max_fill = openems_cfg.get("max_fill_frac")
                    if min_fill is not None and (fill_frac is None or fill_frac < float(min_fill)):
                        rl_val = None
                    if max_fill is not None and (fill_frac is None or fill_frac > float(max_fill)):
                        rl_val = None
                    min_bw_frac = openems_cfg.get("min_bw_frac")
                    bw_val = eval_entry.get(bw_key)
                    if min_bw_frac is not None:
                        if bw_val is None or bw_val < float(min_bw_frac):
                            rl_val = None
                    should_launch = False
                    reason = None
                    if rl_val is not None and rl_val >= min_rl:
                        should_launch = True
                        reason = "rl"
                    else:
                        try:
                            validation_interval = int(openems_cfg.get("validation_interval", 0) or 0)
                        except (TypeError, ValueError):
                            validation_interval = 0
                        try:
                            validation_min_epoch = int(openems_cfg.get("validation_min_epoch", 0) or 0)
                        except (TypeError, ValueError):
                            validation_min_epoch = 0
                        if (
                            validation_interval > 0
                            and (epoch + 1) >= validation_min_epoch
                            and (epoch + 1) % validation_interval == 0
                        ):
                            should_launch = True
                            reason = "validation"
                    if should_launch:
                        openems_epochs = entry.setdefault("openems_epochs", [])
                        if epoch not in openems_epochs:
                            fdtdx_eval_dir = os.path.join(cpu_run_root, out_dir)
                            if eval_entry.get("valid") is False:
                                reasons = eval_entry.get("invalid_reasons") or []
                                if not allow_invalid:
                                    log(f"{run_id} openEMS skipped (invalid cpu eval)")
                                    continue
                                if hard_invalid_reasons and any(r in hard_invalid_reasons for r in reasons):
                                    log(
                                        f"{run_id} openEMS skipped (hard invalid cpu eval: {reasons})"
                                    )
                                    continue
                                log(
                                    f"{run_id} openEMS launching despite invalid cpu eval ({reasons})"
                                )
                            if not os.path.exists(os.path.join(fdtdx_eval_dir, "geometry.npz")):
                                log(f"{run_id} openEMS skipped (missing geometry in {fdtdx_eval_dir})")
                                continue
                            _export_fdtdx_geometry(
                                python_bin,
                                fdtdx_eval_dir,
                                float(eval_entry.get("threshold", thr_val)),
                                log,
                            )
                            log_name = _sanitize_name(run_id)
                            log_path = os.path.join(
                                repo_root,
                                f"nohup_openems_{log_name}_e{epoch + 1:04d}.out",
                            )
                            pid_openems = _launch_openems(
                                python_bin,
                                fdtdx_eval_dir,
                                openems_cfg,
                                log_path,
                            )
                            if pid_openems:
                                openems_epochs.append(epoch)
                                entry.setdefault("openems_pids", []).append(pid_openems)
                                log(
                                    f"{run_id} openEMS metrics started pid={pid_openems} "
                                    f"epoch={epoch} reason={reason}"
                                )
                                active_openems.append(pid_openems)
                if is_default_thr:
                    if (
                        cpu_stop_below is not None
                        and epoch + 1 >= cpu_stop_after
                    ):
                        rl_val = eval_entry.get("rl_min_db")
                        if rl_val is not None and rl_val < float(cpu_stop_below):
                            pid = entry.get("pid")
                            if pid and _pid_alive(pid):
                                os.kill(pid, 15)
                                log(f"{run_id} cpu auto-stop rl_min={rl_val:.2f}")
                    if (
                        cpu_stop_density_after > 0
                        and epoch + 1 >= cpu_stop_density_after
                        and (cpu_stop_density_min is not None or cpu_stop_density_max is not None)
                    ):
                        fill_frac = eval_entry.get("fill_frac")
                        out_of_range = False
                        if fill_frac is None:
                            out_of_range = True
                        if cpu_stop_density_min is not None and fill_frac is not None:
                            if fill_frac < float(cpu_stop_density_min):
                                out_of_range = True
                        if cpu_stop_density_max is not None and fill_frac is not None:
                            if fill_frac > float(cpu_stop_density_max):
                                out_of_range = True
                        if out_of_range:
                            pid = entry.get("pid")
                            if pid and _pid_alive(pid):
                                os.kill(pid, 15)
                                log(
                                    f"{run_id} cpu auto-stop fill_frac={fill_frac} "
                                    f"outside [{cpu_stop_density_min},{cpu_stop_density_max}]"
                                )
                    if cpu_stale_limit is not None:
                        try:
                            stale_limit = int(cpu_stale_limit)
                        except (TypeError, ValueError):
                            stale_limit = 0
                        if stale_limit > 0:
                            last_hash = entry.get("last_cpu_hash")
                            if out_dir == last_hash:
                                entry["stale_count"] = int(entry.get("stale_count", 0)) + 1
                            else:
                                entry["stale_count"] = 0
                                entry["last_cpu_hash"] = out_dir
                            if entry.get("stale_count", 0) >= stale_limit:
                                pid = entry.get("pid")
                                if pid and _pid_alive(pid):
                                    os.kill(pid, 15)
                                    log(
                                        f"{run_id} cpu auto-stop stale binarized hash {out_dir} "
                                        f"({entry['stale_count']} repeats)"
                                    )

        for run in runs:
            run_id = _run_id(run)
            entry = state["runs"][run_id]
            if entry.get("status") == "running":
                pid = entry.get("pid")
                if pid and not _pid_alive(pid):
                    entry["status"] = "done"
                    log(f"{run_id} finished (pid {pid})")
                if entry.get("opt_run") is None and entry.get("start_time"):
                    opt_run = _find_opt_run(
                        opt_root,
                        str(run.get("model")),
                        int(run.get("seed", 0)),
                        float(entry["start_time"]),
                        run.get("init_mode"),
                        run.get("run_tag"),
                    )
                    if opt_run:
                        entry["opt_run"] = opt_run
                        log(f"{run_id} opt_run={opt_run}")
            if (
                entry.get("status") == "running"
                and _pid_alive(entry.get("pid"))
                and entry.get("opt_run")
                and cpu_interval > 0
            ):
                last_by_thr = entry.setdefault("last_cpu_eval_by_threshold", {})
                scheduled = False
                for thr in cpu_thresholds:
                    thr_key = _threshold_key(thr)
                    last_epoch = last_by_thr.get(thr_key)
                    if last_epoch is None:
                        if abs(thr - cpu_threshold) <= 1e-6:
                            last_epoch = entry.get("last_cpu_eval_epoch", -1)
                        else:
                            last_epoch = -1
                    epoch = _cpu_eval_ready(entry["opt_run"], cpu_interval, int(last_epoch))
                    if epoch is None:
                        continue
                    if (run_id, thr_key) in cpu_inflight_runs:
                        continue
                    if cpu_pool is None or len(cpu_jobs) >= cpu_max_inflight:
                        break
                    cpu_port_reference = cpu_cfg.get("port_reference")
                    if not cpu_port_reference and cpu_rl_mode == "thevenin":
                        cpu_port_reference = _default_port_reference(str(run.get("model")))
                    pid = entry.get("pid")
                    try:
                        future = cpu_pool.submit(
                            tw._evaluate_once,
                            entry["opt_run"],
                            str(run.get("model")),
                            epoch,
                            float(thr),
                            cpu_quality,
                            cpu_run_root,
                            cpu_rl_mode,
                            cpu_port_reference,
                            cpu_port_calibration,
                            cpu_calibration_metric,
                            cpu_params_resample,
                            cpu_params_resample_method,
                        )
                        cpu_jobs[future] = {
                            "run_id": run_id,
                            "epoch": epoch,
                            "pid": pid,
                            "threshold": float(thr),
                            "thr_key": thr_key,
                        }
                        cpu_inflight_runs.add((run_id, thr_key))
                        scheduled = True
                    except Exception as exc:
                        log(f"{run_id} cpu eval failed: {exc}")
                    if scheduled:
                        break
            if (
                cpu_eval_on_done
                and entry.get("status") == "done"
                and entry.get("opt_run")
                and cpu_interval > 0
            ):
                last_by_thr = entry.setdefault("last_cpu_eval_by_threshold", {})
                scheduled = False
                for thr in cpu_thresholds:
                    thr_key = _threshold_key(thr)
                    last_epoch = last_by_thr.get(thr_key)
                    if last_epoch is None:
                        if abs(thr - cpu_threshold) <= 1e-6:
                            last_epoch = entry.get("last_cpu_eval_epoch", -1)
                        else:
                            last_epoch = -1
                    epoch = _cpu_eval_ready(entry["opt_run"], cpu_interval, int(last_epoch))
                    if epoch is None:
                        continue
                    if (run_id, thr_key) in cpu_inflight_runs:
                        continue
                    if cpu_pool is None or len(cpu_jobs) >= cpu_max_inflight:
                        break
                    try:
                        cpu_port_reference = cpu_cfg.get("port_reference")
                        if not cpu_port_reference and cpu_rl_mode == "thevenin":
                            cpu_port_reference = _default_port_reference(str(run.get("model")))
                        future = cpu_pool.submit(
                            tw._evaluate_once,
                            entry["opt_run"],
                            str(run.get("model")),
                            epoch,
                            float(thr),
                            cpu_quality,
                            cpu_run_root,
                            cpu_rl_mode,
                            cpu_port_reference,
                            cpu_port_calibration,
                            cpu_calibration_metric,
                            cpu_params_resample,
                            cpu_params_resample_method,
                        )
                        cpu_jobs[future] = {
                            "run_id": run_id,
                            "epoch": epoch,
                            "pid": entry.get("pid"),
                            "threshold": float(thr),
                            "thr_key": thr_key,
                        }
                        cpu_inflight_runs.add((run_id, thr_key))
                        scheduled = True
                    except Exception as exc:
                        log(f"{run_id} cpu eval failed: {exc}")
                    if scheduled:
                        break

        if openems_cfg.get("enabled", False) and scan_runs:
            scan_jobs = state.setdefault("openems_scan_jobs", {})
            last_scan = float(state.get("openems_last_scan", 0.0))
            if _now() - last_scan >= scan_interval:
                index = _index_openems_runs(openems_root)
                openems_done = index["done"]
                openems_pending = index["pending"]
                launched = 0
                scan_order = str(openems_cfg.get("scan_order", "oldest")).strip().lower()
                for root in scan_roots:
                    run_dirs = _collect_fdtdx_runs(root)
                    if scan_order == "newest":
                        run_dirs = list(reversed(run_dirs))
                    for run_dir in run_dirs:
                        run_dir = os.path.abspath(run_dir)
                        if run_dir in openems_done:
                            scan_jobs.setdefault(run_dir, {"status": "done", "openems_run": openems_done[run_dir]})
                            continue
                        if run_dir in openems_pending:
                            scan_jobs.setdefault(run_dir, {"status": "pending", "openems_run": openems_pending[run_dir]})
                            continue
                        if run_dir in scan_jobs and scan_jobs[run_dir].get("status") == "running":
                            continue
                        if max_openems > 0 and len(active_openems) >= max_openems:
                            break
                        log_name = _sanitize_name(os.path.basename(run_dir))
                        log_path_scan = os.path.join(repo_root, f"nohup_openems_scan_{log_name}.out")
                        _export_fdtdx_geometry(
                            python_bin,
                            run_dir,
                            float(openems_threshold),
                            log,
                        )
                        pid_openems = _launch_openems(python_bin, run_dir, openems_cfg, log_path_scan)
                        if pid_openems:
                            scan_jobs[run_dir] = {
                                "status": "running",
                                "pid": pid_openems,
                                "start_time": _now(),
                            }
                            active_openems.append(pid_openems)
                            launched += 1
                        if launched >= scan_max_per_loop:
                            break
                state["openems_last_scan"] = _now()

        if openems_log_interval > 0:
            last_openems_log = float(state.get("openems_last_log", 0.0))
            if _now() - last_openems_log >= openems_log_interval:
                logged = set(state.get("openems_logged", []))
                newly_logged = _log_openems_results(
                    results_log,
                    openems_root,
                    logged,
                    scoring_cfg,
                    state.get("cpu_run_index", {}),
                    log,
                )
                if newly_logged:
                    state["openems_logged"] = list(logged)
                state["openems_last_log"] = _now()

        if archive_enabled:
            last_archive = float(state.get("last_archive", 0.0))
            if _now() - last_archive >= archive_interval:
                _run_archive_passes(
                    python_bin,
                    repo_root,
                    openems_root,
                    archive_out_root,
                    archive_cfg,
                    log,
                )
                state["last_archive"] = _now()

        hybrids_added = _maybe_generate_hybrids(queue, state, args.queue, repo_root, log)
        if hybrids_added:
            runs = queue.get("runs", [])

        for run in runs:
            run_id = _run_id(run)
            entry = state["runs"][run_id]
            if entry.get("status") != "pending":
                continue
            gpu = run.get("gpu")
            if gpu is None:
                for idx in sorted(gpu_status.keys()):
                    if idx in busy_gpus:
                        continue
                    if _gpu_idle(idx, gpu_status):
                        gpu = idx
                        break
            if gpu is None:
                continue
            if int(gpu) in busy_gpus or not _gpu_idle(int(gpu), gpu_status):
                continue

            model_name = str(run.get("model") or "")
            rl_mode = _resolve_rl_mode(run, defaults)
            port_reference = _resolve_port_reference(run, defaults)
            if rl_mode == "thevenin":
                if port_reference:
                    if not os.path.exists(port_reference):
                        log(f"{run_id} missing port_reference={port_reference}; skip")
                        continue
                else:
                    ready, started = _ensure_port_reference(
                        repo_root,
                        python_bin,
                        model_name,
                        int(gpu),
                        state,
                        log,
                    )
                    if started:
                        busy_gpus.add(int(gpu))
                    if not ready:
                        continue

            log_name = _sanitize_name(run_id)
            log_path_run = os.path.join(repo_root, f"nohup_{log_name}.out")
            run_root = run.get("run_root", run_root_default)
            if isinstance(run_root, str) and not os.path.isabs(run_root):
                run_root = os.path.abspath(os.path.join(repo_root, run_root))
            cmd = _build_command(run, defaults, python_bin, run_root)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            with open(log_path_run, "a", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    stdout=handle,
                    stderr=handle,
                    env=env,
                    start_new_session=True,
                )
            entry.update(
                {
                    "status": "running",
                    "pid": proc.pid,
                    "start_time": _now(),
                    "gpu": gpu,
                    "log_path": log_path_run,
                }
            )
            busy_gpus.add(int(gpu))
            log(f"{run_id} started on GPU{gpu} pid={proc.pid}")

        state["last_update"] = _now()
        _save_json(state_path, state)
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())

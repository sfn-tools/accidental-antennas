"""Baseline openEMS validation suite (microstrip + patch)."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import shutil
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

from sim import model_common as mc
from sim import post


MICROSTRIP_DEFAULTS = {
    "f0_hz": 5.0e9,
    "fmin_hz": 2.0e9,
    "fmax_hz": 8.0e9,
    "points": 121,
    "substrate_eps": 4.4,
    "substrate_thickness_mm": 1.6,
    "substrate_len_mm": 24.0,
    "substrate_w_mm": 16.0,
    "line_len_mm": 10.0,
    "line_w_mm": 2.4,
    "load_len_mm": 3.0,
    "port_len_mm": 2.0,
    "port_gap_mm": 2.0,
    "resolution_mm": 1.0,
    "volume_z_mm": 20.0,
    "load_sigma": 3.3,
    "copper_thickness_mm": 0.035,
    "msl_feed_shift_mm": 0.5,
    "msl_measplane_shift_mm": 1.0,
    "msl_feed_r_ohm": None,
    "msl_term_r_ohm": 50.0,
    "msl_excite": -1,
    "s11_inc_min_frac": 0.8,
    "port_r_ohm": 1.0e6,
    "end_criteria": 1.0e-4,
    "nr_ts": 20000,
    "microstrip_smooth_mesh": True,
}


PATCH_DEFAULTS = {
    "f0_hz": 2.4e9,
    "fc_hz": 1.2e9,
    "substrate_eps": 4.4,
    "substrate_kappa": 0.0,
    "substrate_thickness_mm": 1.6,
    "patch_w_mm": 32.0,
    "patch_l_mm": 40.0,
    "substrate_w_mm": 60.0,
    "substrate_l_mm": 60.0,
    "feed_pos_mm": -6.0,
    "feed_r_ohm": 50.0,
    "sim_box_mm": (140.0, 140.0, 120.0),
    "nf2ff_theta_step_deg": 5.0,
    "nf2ff_phi_deg": (0.0, 90.0),
    "end_criteria": 1.0e-4,
    "nr_ts": 20000,
}


PLATE_DEFAULTS = {
    "f0_hz": 1.0e9,
    "fmin_hz": 0.4e9,
    "fmax_hz": 1.6e9,
    "points": 121,
    "plate_size_mm": (6.0, 6.0, 1.0),
    "gap_size_mm": (4.0, 4.0, 2.0),
    "volume_size_mm": (20.0, 20.0, 20.0),
    "resolution_mm": 1.0,
    "load_sigma": 2.5,
    "port_r_ohm": 1.0e6,
    "end_criteria": 1.0e-4,
    "nr_ts": 20000,
}


def _mesh_lines(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    if count <= 0:
        raise ValueError("Mesh step too large for span")
    return np.linspace(start, stop, count + 1)


def _maybe_bootstrap_openems() -> None:
    if importlib.util.find_spec("openEMS") and importlib.util.find_spec("CSXCAD"):
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = []
    for rel in [
        ("openEMS-Project", "openEMS", "python", "build"),
        ("openEMS-Project", "CSXCAD", "python", "build"),
    ]:
        base = os.path.join(repo_root, *rel)
        if not os.path.isdir(base):
            continue
        candidates.extend(glob.glob(os.path.join(base, "lib.*")))
    env_paths = os.environ.get("OPENEMS_PYTHONPATH")
    if env_paths:
        candidates.extend([p for p in env_paths.split(os.pathsep) if p])
    for path in candidates:
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _ensure_openems() -> None:
    _maybe_bootstrap_openems()
    mc.require_openems()


def _safe_float(val: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _save_metrics(path: str, metrics: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def _prune_run(run_dir: str, keep: Iterable[str]) -> None:
    keep_set = set(keep)
    for root, dirs, files in os.walk(run_dir, topdown=False):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), run_dir)
            if rel in keep_set:
                continue
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            path = os.path.join(root, name)
            try:
                if not os.listdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


def _microstrip_geometry(cfg: Dict, load_mode: str, port_type: str):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS import ports

    if load_mode not in {"open", "short", "match"}:
        raise ValueError(f"Unknown load mode: {load_mode}")
    if port_type not in {"lumped", "msl"}:
        raise ValueError(f"Unknown port type: {port_type}")

    f0 = float(cfg["f0_hz"])
    fc = 0.6 * f0
    res = float(cfg["resolution_mm"])
    sub_t = float(cfg["substrate_thickness_mm"])
    cu_thk = float(cfg.get("copper_thickness_mm", 0.0))
    sub_len = float(cfg["substrate_len_mm"])
    sub_w = float(cfg["substrate_w_mm"])
    vol_z = float(cfg.get("volume_z_mm", 12.0))
    line_len = float(cfg["line_len_mm"])
    line_w = float(cfg["line_w_mm"])
    load_len = float(cfg["load_len_mm"])
    port_len = float(cfg["port_len_mm"])
    load_sigma = float(cfg["load_sigma"])

    x0 = -sub_len / 2.0
    x1 = sub_len / 2.0
    y0 = -sub_w / 2.0
    y1 = sub_w / 2.0

    line_x0 = x0 + 2.0
    line_x1 = line_x0 + line_len
    line_y0 = -line_w / 2.0
    line_y1 = line_w / 2.0

    port_x0 = line_x0
    port_x1 = port_x0 + port_len
    port_xc = 0.5 * (port_x0 + port_x1)

    load_x1 = line_x1
    load_x0 = load_x1 - load_len
    z_min = -0.5 * vol_z
    z_max = 0.5 * vol_z
    z_sub0 = -0.5 * sub_t
    z_sub1 = 0.5 * sub_t
    z_gnd0 = z_sub0 - cu_thk
    z_gnd1 = z_sub0
    z_line0 = z_sub1
    z_line1 = z_sub1 + cu_thk

    end_criteria = float(cfg.get("end_criteria", 1.0e-4))
    nr_ts = int(cfg.get("nr_ts", 9000))
    FDTD = openEMS(EndCriteria=end_criteria, NrTS=nr_ts)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.AddLine("x", _mesh_lines(x0 - 3.0, x1 + 3.0, res))
    grid.AddLine("y", _mesh_lines(y0 - 2.0, y1 + 2.0, res))
    grid.AddLine("z", _mesh_lines(z_min, z_max, res))
    # Ensure lumped port center lies on the mesh so the excitation is active.
    grid.AddLine("x", [line_x0, line_x1, port_x0, port_x1, port_xc, load_x0, load_x1])
    grid.AddLine("y", [line_y0, line_y1])
    grid.AddLine("z", [z_gnd0, z_gnd1, z_sub0, z_sub1, z_line0, z_line1])

    sub = CSX.AddMaterial("FR4")
    sub.SetMaterialProperty(epsilon=float(cfg["substrate_eps"]))
    sub.AddBox([x0, y0, z_sub0], [x1, y1, z_sub1], priority=0)

    gnd = CSX.AddMetal("gnd")
    gnd.AddBox([x0, y0, z_gnd0], [x1, y1, z_gnd1], priority=10)
    FDTD.AddEdges2Grid(dirs="xy", properties=gnd)

    top = CSX.AddMetal("top")
    top.AddBox([line_x0, line_y0, z_line0], [line_x1, line_y1, z_line1], priority=10)
    FDTD.AddEdges2Grid(dirs="xy", properties=top)

    if load_mode == "short":
        if port_type == "msl":
            short_r = CSX.AddLumpedElement("ShortR", ny=2, caps=True, R=1.0e-6)
            short_r.AddBox([load_x0, line_y0, z_gnd1], [load_x1, line_y1, z_line0], priority=5)
        else:
            top.AddBox([load_x0, line_y0, z_gnd0], [load_x1, line_y1, z_line1], priority=10)
            FDTD.AddEdges2Grid(dirs="xy", properties=top)
    elif load_mode == "match" and port_type == "msl":
        term_r = cfg.get("msl_term_r_ohm")
        if term_r is not None:
            load_r = CSX.AddLumpedElement("LoadR", ny=2, caps=True, R=float(term_r))
            load_r.AddBox([load_x0, line_y0, z_gnd1], [load_x1, line_y1, z_line0], priority=5)
    elif load_mode == "match" and port_type != "msl":
        load = CSX.AddMaterial("Load")
        load.SetMaterialProperty(epsilon=1.0, kappa=load_sigma)
        load.AddBox([load_x0, line_y0, z_gnd1], [load_x1, line_y1, z_line1], priority=5)

    if cfg.get("microstrip_smooth_mesh", True):
        grid.SmoothMeshLines("x", res, 1.4)
        grid.SmoothMeshLines("y", res, 1.4)
        grid.SmoothMeshLines("z", res, 1.4)

    ports_out = []
    if port_type == "msl":
        port_kwargs = {"priority": 10}
        feed_shift = cfg.get("msl_feed_shift_mm")
        meas_shift = cfg.get("msl_measplane_shift_mm")
        feed_r = cfg.get("msl_feed_r_ohm")
        if feed_shift is not None:
            port_kwargs["FeedShift"] = float(feed_shift)
        if meas_shift is not None:
            port_kwargs["MeasPlaneShift"] = float(meas_shift)
        if feed_r is not None:
            port_kwargs["Feed_R"] = float(feed_r)
        port = ports.MSLPort(
            CSX,
            port_nr=1,
            metal_prop=top,
            start=[line_x0, line_y0, z_line0],
            stop=[port_x1, line_y1, z_gnd1],
            prop_dir="x",
            exc_dir="z",
            excite=int(cfg.get("msl_excite", -1)),
            **port_kwargs,
        )
        ports_out.append(port)
        # Match termination for MSL is handled via a lumped load at the line end.
    else:
        port = ports.LumpedPort(
            CSX,
            port_nr=1,
            R=float(cfg.get("port_r_ohm", 50.0)),
            start=[port_xc, 0.0, z_gnd0],
            stop=[port_xc, 0.0, z_line1],
            exc_dir="z",
            excite=1,
        )
        ports_out.append(port)

    meta = {
        "port_type": port_type,
        "load_mode": load_mode,
        "line_x0": line_x0,
        "line_x1": line_x1,
        "port_x0": port_x0,
        "port_x1": port_x1,
        "load_x0": load_x0,
        "load_x1": load_x1,
    }
    return FDTD, ports_out, meta


def _parallel_plate_geometry(cfg: Dict, load_mode: str):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS, ports

    if load_mode not in {"open", "short", "match"}:
        raise ValueError(f"Unknown load mode: {load_mode}")

    f0 = float(cfg["f0_hz"])
    fc = 0.6 * f0
    res = float(cfg["resolution_mm"])
    vol_x, vol_y, vol_z = cfg["volume_size_mm"]
    plate_x, plate_y, plate_z = cfg["plate_size_mm"]
    gap_x, gap_y, gap_z = cfg["gap_size_mm"]

    end_criteria = float(cfg.get("end_criteria", 1.0e-4))
    nr_ts = int(cfg.get("nr_ts", 9000))
    FDTD = openEMS(EndCriteria=end_criteria, NrTS=nr_ts)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.AddLine("x", [-vol_x / 2.0, vol_x / 2.0])
    grid.AddLine("y", [-vol_y / 2.0, vol_y / 2.0])
    grid.AddLine("z", [-vol_z / 2.0, vol_z / 2.0])
    grid.SmoothMeshLines("x", res, 1.4)
    grid.SmoothMeshLines("y", res, 1.4)
    grid.SmoothMeshLines("z", res, 1.4)

    pec = CSX.AddMetal("pec")
    bottom_z0 = -gap_z / 2.0 - plate_z
    bottom_z1 = -gap_z / 2.0
    top_z0 = gap_z / 2.0
    top_z1 = gap_z / 2.0 + plate_z
    pec.AddBox(
        [-plate_x / 2.0, -plate_y / 2.0, bottom_z0],
        [plate_x / 2.0, plate_y / 2.0, bottom_z1],
        priority=10,
    )
    pec.AddBox(
        [-plate_x / 2.0, -plate_y / 2.0, top_z0],
        [plate_x / 2.0, plate_y / 2.0, top_z1],
        priority=10,
    )
    FDTD.AddEdges2Grid(dirs="xy", properties=pec)

    load = None
    if load_mode == "short":
        pec.AddBox(
            [-gap_x / 2.0, -gap_y / 2.0, -gap_z / 2.0],
            [gap_x / 2.0, gap_y / 2.0, gap_z / 2.0],
            priority=10,
        )
    elif load_mode == "match":
        load = CSX.AddMaterial("Load")
        load.SetMaterialProperty(epsilon=1.0, kappa=float(cfg["load_sigma"]))
        load.AddBox(
            [-gap_x / 2.0, -gap_y / 2.0, -gap_z / 2.0],
            [gap_x / 2.0, gap_y / 2.0, gap_z / 2.0],
            priority=5,
        )

    port = ports.LumpedPort(
        CSX,
        port_nr=1,
        R=float(cfg.get("port_r_ohm", 50.0)),
        start=[0.0, 0.0, -gap_z / 2.0],
        stop=[0.0, 0.0, gap_z / 2.0],
        exc_dir="z",
        excite=1,
        priority=5,
    )

    meta = {"load_mode": load_mode}
    return FDTD, port, meta


def run_microstrip_case(
    cfg: Dict,
    load_mode: str,
    port_type: str,
    out_dir: str,
    prune: bool,
) -> Dict:
    _ensure_openems()
    from openEMS import ports

    os.makedirs(out_dir, exist_ok=True)
    FDTD, ports_out, meta = _microstrip_geometry(cfg, load_mode, port_type)
    FDTD.Run(out_dir, cleanup=False, verbose=0)

    freq = np.linspace(float(cfg["fmin_hz"]), float(cfg["fmax_hz"]), int(cfg["points"]))
    for port in ports_out:
        port.CalcPort(out_dir, freq, ref_impedance=50.0)

    port0 = ports_out[0]
    uf_tot = port0.uf_tot
    if_tot = port0.if_tot
    if_tot = np.where(np.abs(if_tot) < 1e-30, 1e-30 + 0j, if_tot)
    zin = uf_tot / if_tot
    s11_base = (zin - 50.0) / (zin + 50.0)
    uf_ref = getattr(port0, "uf_ref", None)
    uf_inc = getattr(port0, "uf_inc", None)
    valid = None
    s11_wave = None
    if uf_ref is not None and uf_inc is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            s11_wave = np.where(np.abs(uf_inc) > 1e-30, uf_ref / uf_inc, s11_base)
        inc_frac = float(cfg.get("s11_inc_min_frac", 0.0) or 0.0)
        if inc_frac > 0.0:
            inc_mag = np.abs(uf_inc)
            inc_max = float(np.max(inc_mag)) if inc_mag.size else 0.0
            if inc_max > 0.0:
                valid = inc_mag >= inc_frac * inc_max
            else:
                valid = np.ones_like(inc_mag, dtype=bool)
        else:
            valid = np.ones_like(uf_inc, dtype=bool)
        s11 = np.where(valid, s11_wave, s11_base)
        s11 = np.where(np.isfinite(s11), s11, s11_base)
    else:
        s11 = s11_base
    s11_db = post.s11_db(s11)
    rl_db = -s11_db

    post.save_s11_csv(os.path.join(out_dir, "s11.csv"), freq, s11_db, zin)
    post.save_s11_plot(os.path.join(out_dir, "s11.png"), freq, s11_db)
    post.save_smith_csv(os.path.join(out_dir, "s11_smith.csv"), freq, s11)
    if s11_wave is not None:
        s11_wave_masked = np.where(valid, s11_wave, np.nan + 1j * np.nan) if valid is not None else s11_wave
        post.save_smith_csv(os.path.join(out_dir, "s11_wave.csv"), freq, s11_wave_masked)

    f0 = float(cfg["f0_hz"])
    idx_f0 = int(np.argmin(np.abs(freq - f0))) if len(freq) else 0
    if valid is not None and s11_wave is not None and valid.size:
        if valid[idx_f0]:
            s11_f0_db = float(post.s11_db(np.asarray([s11_wave[idx_f0]]))[0])
            rl_f0_db = float(-s11_f0_db)
        elif np.any(valid):
            valid_idx = np.where(valid)[0]
            nearest = valid_idx[np.argmin(np.abs(freq[valid_idx] - f0))]
            s11_f0_db = float(post.s11_db(np.asarray([s11_wave[nearest]]))[0])
            rl_f0_db = float(-s11_f0_db)
        else:
            s11_f0_db = float(np.interp(f0, freq, s11_db))
            rl_f0_db = float(np.interp(f0, freq, rl_db))
    else:
        s11_f0_db = float(np.interp(f0, freq, s11_db))
        rl_f0_db = float(np.interp(f0, freq, rl_db))
    zin_f0 = np.interp(f0, freq, zin.real) + 1j * np.interp(f0, freq, zin.imag)

    metrics = {
        "case": "microstrip",
        "port_type": port_type,
        "load_mode": load_mode,
        "f0_hz": f0,
        "s11_f0_db": s11_f0_db,
        "rl_f0_db": rl_f0_db,
        "rl_min_db": float(np.min(-post.s11_db(s11_wave)[valid])) if valid is not None and s11_wave is not None and np.any(valid) else float(np.min(rl_db)),
        "s11_valid_frac": float(np.sum(valid) / len(valid)) if valid is not None and valid.size else float("nan"),
        "zin_f0_real_ohm": _safe_float(np.real(zin_f0)),
        "zin_f0_imag_ohm": _safe_float(np.imag(zin_f0)),
        "freq_min_hz": float(freq[0]),
        "freq_max_hz": float(freq[-1]),
        "meta": meta,
    }
    _save_metrics(os.path.join(out_dir, "metrics.json"), metrics)

    if prune:
        _prune_run(out_dir, keep={"metrics.json", "s11.csv", "s11.png", "s11_smith.csv", "s11_wave.csv"})
    return metrics


def run_plate_case(cfg: Dict, load_mode: str, out_dir: str, prune: bool) -> Dict:
    _ensure_openems()
    os.makedirs(out_dir, exist_ok=True)
    FDTD, port, meta = _parallel_plate_geometry(cfg, load_mode)
    FDTD.Run(out_dir, cleanup=False, verbose=0)

    freq = np.linspace(float(cfg["fmin_hz"]), float(cfg["fmax_hz"]), int(cfg["points"]))
    port.CalcPort(out_dir, freq, ref_impedance=50.0)

    uf_tot = port.uf_tot
    if_tot = port.if_tot
    if_tot = np.where(np.abs(if_tot) < 1e-30, 1e-30 + 0j, if_tot)
    zin = uf_tot / if_tot
    s11_base = (zin - 50.0) / (zin + 50.0)
    uf_ref = getattr(port, "uf_ref", None)
    uf_inc = getattr(port, "uf_inc", None)
    valid = None
    s11_wave = None
    if uf_ref is not None and uf_inc is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            s11_wave = np.where(np.abs(uf_inc) > 1e-30, uf_ref / uf_inc, s11_base)
        inc_frac = float(cfg.get("s11_inc_min_frac", 0.0) or 0.0)
        if inc_frac > 0.0:
            inc_mag = np.abs(uf_inc)
            inc_max = float(np.max(inc_mag)) if inc_mag.size else 0.0
            if inc_max > 0.0:
                valid = inc_mag >= inc_frac * inc_max
            else:
                valid = np.ones_like(inc_mag, dtype=bool)
        else:
            valid = np.ones_like(uf_inc, dtype=bool)
        s11 = np.where(valid, s11_wave, s11_base)
        s11 = np.where(np.isfinite(s11), s11, s11_base)
    else:
        s11 = s11_base
    s11_db = post.s11_db(s11)
    rl_db = -s11_db

    post.save_s11_csv(os.path.join(out_dir, "s11.csv"), freq, s11_db, zin)
    post.save_s11_plot(os.path.join(out_dir, "s11.png"), freq, s11_db)
    post.save_smith_csv(os.path.join(out_dir, "s11_smith.csv"), freq, s11)

    f0 = float(cfg["f0_hz"])
    idx_f0 = int(np.argmin(np.abs(freq - f0))) if len(freq) else 0
    if valid is not None and s11_wave is not None and valid.size:
        if valid[idx_f0]:
            s11_f0_db = float(post.s11_db(np.asarray([s11_wave[idx_f0]]))[0])
            rl_f0_db = float(-s11_f0_db)
        elif np.any(valid):
            valid_idx = np.where(valid)[0]
            nearest = valid_idx[np.argmin(np.abs(freq[valid_idx] - f0))]
            s11_f0_db = float(post.s11_db(np.asarray([s11_wave[nearest]]))[0])
            rl_f0_db = float(-s11_f0_db)
        else:
            s11_f0_db = float(np.interp(f0, freq, s11_db))
            rl_f0_db = float(np.interp(f0, freq, rl_db))
    else:
        s11_f0_db = float(np.interp(f0, freq, s11_db))
        rl_f0_db = float(np.interp(f0, freq, rl_db))
    zin_f0 = np.interp(f0, freq, zin.real) + 1j * np.interp(f0, freq, zin.imag)

    metrics = {
        "case": "plate",
        "load_mode": load_mode,
        "f0_hz": f0,
        "s11_f0_db": s11_f0_db,
        "rl_f0_db": rl_f0_db,
        "rl_min_db": float(np.min(-post.s11_db(s11_wave)[valid])) if valid is not None and s11_wave is not None and np.any(valid) else float(np.min(rl_db)),
        "s11_valid_frac": float(np.sum(valid) / len(valid)) if valid is not None and valid.size else float("nan"),
        "zin_f0_real_ohm": _safe_float(np.real(zin_f0)),
        "zin_f0_imag_ohm": _safe_float(np.imag(zin_f0)),
        "freq_min_hz": float(freq[0]),
        "freq_max_hz": float(freq[-1]),
        "meta": meta,
    }
    _save_metrics(os.path.join(out_dir, "metrics.json"), metrics)

    if prune:
        _prune_run(out_dir, keep={"metrics.json", "s11.csv", "s11.png", "s11_smith.csv"})
    return metrics


def _patch_geometry(cfg: Dict):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    f0 = float(cfg["f0_hz"])
    fc = float(cfg["fc_hz"])
    sim_box = cfg["sim_box_mm"]
    sub_t = float(cfg["substrate_thickness_mm"])

    end_criteria = float(cfg.get("end_criteria", 1.0e-4))
    nr_ts = int(cfg.get("nr_ts", 20000))
    FDTD = openEMS(EndCriteria=end_criteria, NrTS=nr_ts)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)

    mesh_res = 3e8 / (f0 * math.sqrt(cfg["substrate_eps"])) / 20.0 / 1e-3
    mesh_res = max(mesh_res, 0.3)
    grid.AddLine("x", [-sim_box[0] / 2.0, sim_box[0] / 2.0])
    grid.AddLine("y", [-sim_box[1] / 2.0, sim_box[1] / 2.0])
    grid.AddLine("z", [-sim_box[2] / 3.0, sim_box[2] * 2.0 / 3.0])

    patch = CSX.AddMetal("patch")
    patch.AddBox(
        start=[-cfg["patch_w_mm"] / 2.0, -cfg["patch_l_mm"] / 2.0, sub_t],
        stop=[cfg["patch_w_mm"] / 2.0, cfg["patch_l_mm"] / 2.0, sub_t],
        priority=10,
    )
    FDTD.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=mesh_res / 2.0)

    substrate = CSX.AddMaterial("substrate")
    substrate.SetMaterialProperty(epsilon=cfg["substrate_eps"], kappa=cfg["substrate_kappa"])
    substrate.AddBox(
        start=[-cfg["substrate_w_mm"] / 2.0, -cfg["substrate_l_mm"] / 2.0, 0.0],
        stop=[cfg["substrate_w_mm"] / 2.0, cfg["substrate_l_mm"] / 2.0, sub_t],
        priority=0,
    )

    grid.AddLine("z", np.linspace(0.0, sub_t, 5))

    gnd = CSX.AddMetal("gnd")
    gnd.AddBox(
        start=[-cfg["substrate_w_mm"] / 2.0, -cfg["substrate_l_mm"] / 2.0, 0.0],
        stop=[cfg["substrate_w_mm"] / 2.0, cfg["substrate_l_mm"] / 2.0, 0.0],
        priority=10,
    )
    FDTD.AddEdges2Grid(dirs="xy", properties=gnd)

    port = FDTD.AddLumpedPort(
        1,
        cfg["feed_r_ohm"],
        [cfg["feed_pos_mm"], 0.0, 0.0],
        [cfg["feed_pos_mm"], 0.0, sub_t],
        "z",
        1.0,
        priority=5,
        edges2grid="xy",
    )

    grid.SmoothMeshLines("all", mesh_res, 1.4)
    nf2ff = FDTD.CreateNF2FFBox()
    return FDTD, port, nf2ff


def run_patch_case(cfg: Dict, out_dir: str, prune: bool, skip_nf2ff: bool) -> Dict:
    _ensure_openems()
    os.makedirs(out_dir, exist_ok=True)
    FDTD, port, nf2ff = _patch_geometry(cfg)
    FDTD.Run(out_dir, cleanup=False, verbose=0)

    f0 = float(cfg["f0_hz"])
    fc = float(cfg["fc_hz"])
    freq = np.linspace(max(1.0e9, f0 - fc), f0 + fc, 401)
    port.CalcPort(out_dir, freq, ref_impedance=50.0)
    s11, zin = post.port_s11_zin(port)
    s11_db = post.s11_db(s11)
    rl_db = -s11_db

    post.save_s11_csv(os.path.join(out_dir, "s11.csv"), freq, s11_db, zin)
    post.save_s11_plot(os.path.join(out_dir, "s11.png"), freq, s11_db)
    post.save_smith_csv(os.path.join(out_dir, "s11_smith.csv"), freq, s11)

    idx = int(np.argmin(s11_db)) if len(s11_db) else 0
    f_res = float(freq[idx])
    zin_res = zin[idx] if len(zin) else complex("nan")

    metrics = {
        "case": "patch",
        "f_res_hz": f_res,
        "s11_min_db": float(s11_db[idx]) if len(s11_db) else float("nan"),
        "rl_max_db": float(rl_db[idx]) if len(rl_db) else float("nan"),
        "zin_res_real_ohm": _safe_float(np.real(zin_res)),
        "zin_res_imag_ohm": _safe_float(np.imag(zin_res)),
    }

    if not skip_nf2ff:
        theta = np.arange(0.0, 180.0 + cfg["nf2ff_theta_step_deg"], cfg["nf2ff_theta_step_deg"])
        phi = list(cfg["nf2ff_phi_deg"])
        nf2ff_res = nf2ff.CalcNF2FF(out_dir, f_res, theta, phi, center=[0, 0, 0])
        dmax = float(nf2ff_res.Dmax[0]) if hasattr(nf2ff_res, "Dmax") else float("nan")
        metrics["directivity_max_db"] = 10.0 * math.log10(max(dmax, 1e-12)) if np.isfinite(dmax) else float("nan")

    _save_metrics(os.path.join(out_dir, "metrics.json"), metrics)

    if prune:
        _prune_run(out_dir, keep={"metrics.json", "s11.csv", "s11.png", "s11_smith.csv"})
    return metrics


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["microstrip", "patch", "plate", "all"], default="all")
    parser.add_argument("--out-root", default=None, help="Output root folder")
    parser.add_argument("--port-types", default="lumped,msl", help="Comma-separated list for microstrip")
    parser.add_argument("--load-modes", default="open,short,match", help="Comma-separated list for microstrip")
    parser.add_argument("--resolution-mm", type=float, default=None, help="Override mesh resolution (microstrip/plate)")
    parser.add_argument("--msl-feed-shift-mm", type=float, default=None, help="Override MSL FeedShift (microstrip)")
    parser.add_argument("--msl-measplane-shift-mm", type=float, default=None, help="Override MSL MeasPlaneShift (microstrip)")
    parser.add_argument("--msl-excite", type=int, default=None, help="Override MSL excite sign (microstrip)")
    parser.add_argument(
        "--no-microstrip-smooth",
        action="store_true",
        help="Disable mesh smoothing for microstrip cases",
    )
    parser.add_argument("--line-len-mm", type=float, default=None, help="Override microstrip line length")
    parser.add_argument("--line-w-mm", type=float, default=None, help="Override microstrip line width")
    parser.add_argument("--port-len-mm", type=float, default=None, help="Override microstrip port length")
    parser.add_argument("--substrate-len-mm", type=float, default=None, help="Override microstrip substrate length")
    parser.add_argument("--substrate-w-mm", type=float, default=None, help="Override microstrip substrate width")
    parser.add_argument("--copper-thickness-mm", type=float, default=None, help="Override microstrip copper thickness")
    parser.add_argument("--port-r-ohm", type=float, default=None, help="Override lumped port resistance (microstrip/plate)")
    parser.add_argument("--nr-ts", type=int, default=None, help="Override FDTD NrTS for microstrip/plate")
    parser.add_argument("--end-criteria", type=float, default=None, help="Override FDTD EndCriteria for microstrip/plate")
    parser.add_argument(
        "--s11-inc-min-frac",
        type=float,
        default=None,
        help="Mask wave S11 where |uf_inc| < frac * max(|uf_inc|) (microstrip)",
    )
    parser.add_argument("--fmin-hz", type=float, default=None, help="Override fmin for microstrip/plate")
    parser.add_argument("--fmax-hz", type=float, default=None, help="Override fmax for microstrip/plate")
    parser.add_argument("--f0-hz", type=float, default=None, help="Override f0 for microstrip/plate")
    parser.add_argument("--points", type=int, default=None, help="Override sweep points for microstrip/plate")
    parser.add_argument("--load-sigma", type=float, default=None, help="Override microstrip load sigma")
    parser.add_argument("--summary", default=None, help="Optional summary JSON output path")
    parser.add_argument("--prune", action="store_true", help="Remove large openEMS artifacts")
    parser.add_argument("--skip-nf2ff", action="store_true", help="Skip NF2FF for patch")
    args = parser.parse_args()

    out_root = args.out_root
    if not out_root:
        out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "baseline_suite"))

    summary: Dict[str, Dict] = {}
    micro_cfg = dict(MICROSTRIP_DEFAULTS)
    plate_cfg = dict(PLATE_DEFAULTS)
    patch_cfg = dict(PATCH_DEFAULTS)
    if args.load_sigma is not None:
        micro_cfg["load_sigma"] = float(args.load_sigma)
    if args.line_len_mm is not None:
        micro_cfg["line_len_mm"] = float(args.line_len_mm)
    if args.line_w_mm is not None:
        micro_cfg["line_w_mm"] = float(args.line_w_mm)
    if args.port_len_mm is not None:
        micro_cfg["port_len_mm"] = float(args.port_len_mm)
    if args.substrate_len_mm is not None:
        micro_cfg["substrate_len_mm"] = float(args.substrate_len_mm)
    if args.substrate_w_mm is not None:
        micro_cfg["substrate_w_mm"] = float(args.substrate_w_mm)
    if args.copper_thickness_mm is not None:
        micro_cfg["copper_thickness_mm"] = float(args.copper_thickness_mm)
    if args.port_r_ohm is not None:
        micro_cfg["port_r_ohm"] = float(args.port_r_ohm)
    if args.nr_ts is not None:
        micro_cfg["nr_ts"] = int(args.nr_ts)
    if args.end_criteria is not None:
        micro_cfg["end_criteria"] = float(args.end_criteria)
    if args.s11_inc_min_frac is not None:
        micro_cfg["s11_inc_min_frac"] = float(args.s11_inc_min_frac)
    if args.msl_feed_shift_mm is not None:
        micro_cfg["msl_feed_shift_mm"] = float(args.msl_feed_shift_mm)
    if args.msl_measplane_shift_mm is not None:
        micro_cfg["msl_measplane_shift_mm"] = float(args.msl_measplane_shift_mm)
    if args.msl_excite is not None:
        micro_cfg["msl_excite"] = int(args.msl_excite)
    if args.no_microstrip_smooth:
        micro_cfg["microstrip_smooth_mesh"] = False
    for cfg in (micro_cfg, plate_cfg):
        if args.resolution_mm is not None:
            cfg["resolution_mm"] = float(args.resolution_mm)
        if args.fmin_hz is not None:
            cfg["fmin_hz"] = float(args.fmin_hz)
        if args.fmax_hz is not None:
            cfg["fmax_hz"] = float(args.fmax_hz)
        if args.f0_hz is not None:
            cfg["f0_hz"] = float(args.f0_hz)
        if args.points is not None:
            cfg["points"] = int(args.points)
        if args.end_criteria is not None:
            cfg["end_criteria"] = float(args.end_criteria)
        if args.nr_ts is not None:
            cfg["nr_ts"] = int(args.nr_ts)
        if args.port_r_ohm is not None:
            cfg["port_r_ohm"] = float(args.port_r_ohm)
    if args.end_criteria is not None:
        patch_cfg["end_criteria"] = float(args.end_criteria)
    if args.nr_ts is not None:
        patch_cfg["nr_ts"] = int(args.nr_ts)
    if args.case in {"microstrip", "all"}:
        for port_type in _split_list(args.port_types):
            for load_mode in _split_list(args.load_modes):
                tag = f"microstrip_{port_type}_{load_mode}"
                run_dir = os.path.join(out_root, tag)
                metrics = run_microstrip_case(
                    micro_cfg,
                    load_mode=load_mode,
                    port_type=port_type,
                    out_dir=run_dir,
                    prune=args.prune,
                )
                summary[tag] = metrics

    if args.case in {"plate", "all"}:
        for load_mode in _split_list(args.load_modes):
            tag = f"plate_{load_mode}"
            run_dir = os.path.join(out_root, tag)
            metrics = run_plate_case(
                plate_cfg,
                load_mode=load_mode,
                out_dir=run_dir,
                prune=args.prune,
            )
            summary[tag] = metrics

    if args.case in {"patch", "all"}:
        tag = "patch_simple"
        run_dir = os.path.join(out_root, tag)
        metrics = run_patch_case(
            patch_cfg,
            out_dir=run_dir,
            prune=args.prune,
            skip_nf2ff=args.skip_nf2ff,
        )
        summary[tag] = metrics

    if args.summary:
        _save_metrics(args.summary, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

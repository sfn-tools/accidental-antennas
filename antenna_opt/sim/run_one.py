"""Run a single openEMS evaluation for one antenna model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

from . import matching
from . import model_common as mc
from . import post
from . import score
from . import model_dir24_quasiyagi
from . import model_dir24_acs_monopole_yagi
from . import model_dir24_acs_ursi_uniplanar
from . import model_dir24_acs_ursi_ground
from . import model_dir5_ms_cps_dipole_yagi

try:
    from . import model_dir5_quasiyagi
    from . import model_omni_dual_ifa
except Exception:
    model_dir5_quasiyagi = None
    model_omni_dual_ifa = None


MODELS = {
    "dir24": model_dir24_quasiyagi,
    "dir24_acs": model_dir24_acs_monopole_yagi,
    "dir24_acs_ursi_uni": model_dir24_acs_ursi_uniplanar,
    "dir24_acs_ursi_gnd": model_dir24_acs_ursi_ground,
    "dir5_ms_cps": model_dir5_ms_cps_dipole_yagi,
}
if model_dir5_quasiyagi is not None:
    MODELS["dir5"] = model_dir5_quasiyagi
if model_omni_dual_ifa is not None:
    MODELS["omni"] = model_omni_dual_ifa


def _merge_params(defaults: Dict, overrides: Dict) -> Dict:
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _load_params(path: str | None) -> Dict:
    if not path:
        return {}
    return mc.load_json(path)


def _resolve_payload(model_name: str | None, payload: Dict, constraints: mc.SliceConstraints) -> Dict:
    if "model" in payload and not model_name:
        model_name = payload["model"]
    if not model_name:
        model_name = "dir24"
    model = MODELS[model_name]
    params_override = payload.get("params", payload)
    params = _merge_params(model.default_params(constraints), params_override)
    substrate = mc.load_substrate(payload.get("substrate"), base=getattr(model, "DEFAULT_SUBSTRATE", None))
    return {
        "model_name": model_name,
        "params": params,
        "substrate": substrate,
    }


def _constraint_penalty(constraints: mc.SliceConstraints, constraint_info: Dict, geom: mc.Geometry) -> float:
    penalty = mc.wedge_violation_penalty(geom, constraints)
    min_trace = constraint_info.get("min_trace_mm", 1e9)
    min_gap = constraint_info.get("min_gap_mm", 1e9)
    if min_trace < constraints.min_trace_mm:
        penalty += constraints.min_trace_mm - min_trace
    if min_gap < constraints.min_gap_mm:
        penalty += constraints.min_gap_mm - min_gap
    if penalty < 1e-6:
        penalty = 0.0
    return penalty


def _build_cluster_geometry(geom: mc.Geometry, constraints: mc.SliceConstraints) -> mc.Geometry:
    angles = [0.0, constraints.slice_angle_deg, -constraints.slice_angle_deg]
    combined = mc.Geometry()
    feed_points = []
    for ang in angles:
        rot = mc.rotate_geometry(geom, ang)
        combined.top_polys.extend(rot.top_polys)
        combined.ground_polys.extend(rot.ground_polys)
        feed_points.append(rot.feed_point)
        if "port_defs" in rot.meta:
            combined.meta.setdefault("port_defs", []).extend(rot.meta["port_defs"])
    combined.meta["feed_points"] = feed_points
    return combined


def _excite_fc(f0: float, fmin: float, fmax: float) -> float:
    return max((fmax - fmin) * 2.0, f0 * 0.25)


def _finite_argmax(values: np.ndarray) -> int | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    idxs = np.where(finite)[0]
    return int(idxs[int(np.argmax(values[finite]))])


def _primary_band_key(model) -> str:
    if "24" in model.BANDS:
        return "24"
    if "5" in model.BANDS:
        return "5"
    return list(model.BANDS.keys())[0]


def _maybe_round(value: float | None, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def _band_s11_invalid(s11_db: np.ndarray, mask: np.ndarray) -> str | None:
    if not np.any(mask):
        return "no_band_points"
    band = s11_db[mask]
    if not np.all(np.isfinite(band)):
        return "non_finite_s11"
    if np.any(band > 0.5):
        return "s11_above_0p5_db"
    return None


def print_candidate_summary(model_name: str, metrics: Dict) -> None:
    f_peak = metrics.get("f_peak_hz")
    rl_peak = metrics.get("rl_peak_db")
    rl_min = metrics.get("rl_min_in_band_db")
    if rl_min is None:
        rl_min = metrics.get("rl_min_db")
    if rl_min is None:
        rl_min = metrics.get("rl_min_24_db")
    if rl_min is None:
        rl_min = metrics.get("rl_min_5_db")
    score_val = metrics.get("score")
    f_peak_ghz = float(f_peak) / 1e9 if f_peak is not None else float("nan")
    rl_peak_val = float(rl_peak) if rl_peak is not None else float("nan")
    rl_min_val = float(rl_min) if rl_min is not None else float("nan")
    score_fmt = float(score_val) if score_val is not None else float("nan")
    print(
        f"[{model_name}] f_peak={f_peak_ghz:.3f} GHz "
        f"RL_peak={rl_peak_val:.2f} dB "
        f"RL_min_in_band={rl_min_val:.2f} dB "
        f"score={score_fmt:.2f}"
    )


def _run_single(
    model_name: str,
    params: Dict,
    substrate: mc.SubstrateConfig,
    constraints: mc.SliceConstraints,
    quality_name: str,
    run_dir: str,
    force: bool,
    stage: str,
) -> Dict:
    model = MODELS[model_name]
    primary_band = _primary_band_key(model)
    geom, constraint_info = model.build_geometry(params, constraints)
    penalty = _constraint_penalty(constraints, constraint_info, geom)
    if penalty > 0:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        mc.save_json(os.path.join(run_dir, "params.json"), {"params": params, "substrate": substrate.__dict__})
        metrics = {
            "model": model_name,
            "valid": False,
            "constraint_penalty": round(penalty, 3),
            "iso_db": 0.0,
        }
        if primary_band == "24":
            metrics["iso_24_db"] = metrics["iso_db"]
        elif primary_band == "5":
            metrics["iso_5_db"] = metrics["iso_db"]
        post.save_metrics(os.path.join(run_dir, "metrics.json"), metrics)
        post.save_summary(os.path.join(run_dir, "summary.txt"), metrics)
        return metrics
    if model_name == "omni":
        fmin = min(b["f_low"] for b in model.BANDS.values())
        fmax = max(b["f_high"] for b in model.BANDS.values())
        f0 = 0.5 * (fmin + fmax)
        band_key = "omni"
    else:
        band_key = list(model.BANDS.keys())[0]
        band = model.BANDS[band_key]
        fmin = band["f_low"]
        fmax = band["f_high"]
        f0 = band["f0"]

    quality = mc.quality_from_name(quality_name, fmax, substrate.eps_r, constraints.min_trace_mm)
    fc = _excite_fc(f0, fmin, fmax)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    mc.save_json(os.path.join(run_dir, "params.json"), {"params": params, "substrate": substrate.__dict__})

    FDTD, ports, nf2ff, meta = mc.build_simulation(
        geom,
        substrate,
        constraints,
        quality,
        f0_hz=f0,
        fc_hz=fc,
        fmax_hz=fmax,
        excite_port=0,
        port_count=1,
    )

    sim_path = run_dir
    if force or not os.path.exists(os.path.join(sim_path, "openEMS_log.txt")):
        FDTD.Run(sim_path, cleanup=force, verbose=0)

    freq_plot = np.linspace(0.1e9, 6.0e9, 901)
    freq_eval_max = 6.0e9
    if "5" in model.BANDS:
        freq_eval_max = 12.0e9
    if freq_eval_max > freq_plot[-1]:
        points = int(round(len(freq_plot) * freq_eval_max / freq_plot[-1]))
        freq_eval = np.linspace(0.1e9, freq_eval_max, points)
    else:
        freq_eval = freq_plot
    post.calc_sparams(ports, sim_path, freq_eval, ref_impedance=50, excite_port=0)
    s11, zin = post.port_s11_zin(ports[0])
    s11_db = post.s11_db(s11)
    if freq_eval is freq_plot:
        s11_db_plot = s11_db
        zin_plot = zin
    else:
        s11_db_plot = np.interp(freq_plot, freq_eval, s11_db)
        zin_real = np.interp(freq_plot, freq_eval, np.real(zin))
        zin_imag = np.interp(freq_plot, freq_eval, np.imag(zin))
        zin_plot = zin_real + 1j * zin_imag

    post.save_s11_csv(os.path.join(sim_path, "s11.csv"), freq_plot, s11_db_plot, zin_plot)
    post.save_s11_plot(os.path.join(sim_path, "s11.png"), freq_plot, s11_db_plot)

    match_meta = None
    s11_matched_db_plot = None
    s11_matched_plot = None
    zin_matched_plot = None
    try:
        f0_idx = int(np.argmin(np.abs(freq_eval - f0)))
        zin_f0 = zin[f0_idx]
        match_meta = matching.calc_l_match(zin_f0, f0)
    except Exception:
        match_meta = None

    if match_meta:
        zin_matched_plot, s11_matched_plot, s11_matched_db_plot = matching.apply_l_match(
            zin_plot, freq_plot, match_meta, z0=50.0
        )
        post.save_s11_csv(
            os.path.join(sim_path, "s11_matched.csv"),
            freq_plot,
            s11_matched_db_plot,
            zin_matched_plot,
        )
        post.save_s11_plot(os.path.join(sim_path, "s11_matched.png"), freq_plot, s11_matched_db_plot)
        post.save_smith_csv(os.path.join(sim_path, "s11_matched_smith.csv"), freq_plot, s11_matched_plot)

    meta_path = os.path.join(sim_path, "meta.json")
    mc.save_json(meta_path, {"nf2ff": meta, "params": params})

    rl_db = -s11_db
    peak_idx = _finite_argmax(rl_db)
    f_peak = float(freq_eval[peak_idx]) if peak_idx is not None else None
    rl_peak = float(rl_db[peak_idx]) if peak_idx is not None else None

    if model_name == "omni":
        metrics = {
            "model": model_name,
            "constraint_penalty": round(penalty, 3),
        }

        invalid_reason = None
        for key, band in model.BANDS.items():
            mask = (freq_eval >= band["f_low"]) & (freq_eval <= band["f_high"])
            invalid = _band_s11_invalid(s11_db, mask)
            if invalid and invalid_reason is None:
                invalid_reason = invalid

            band_rl = rl_db[mask]
            finite = np.isfinite(band_rl)
            rl_min = float(np.min(band_rl[finite])) if np.any(finite) else None
            metrics[f"rl_min_{key}_db"] = _maybe_round(rl_min, 3)

            phi, gains = post.calc_plane_pattern(
                nf2ff,
                sim_path,
                band["f0"],
                range(0, 361, 10),
                outfile=f"nf2ff_plane_{key}.h5",
            )
            post.save_pattern_csv(os.path.join(sim_path, f"pattern_{key}.csv"), phi, gains)
            ripple = float(np.max(gains) - np.min(gains)) if len(gains) else 0.0
            metrics[f"ripple_{key}_db"] = round(ripple, 3)
            metrics[f"gain_avg_{key}_dbi"] = round(float(np.mean(gains)), 3)

        metrics["f_peak_hz"] = _maybe_round(f_peak, 1)
        metrics["rl_peak_db"] = _maybe_round(rl_peak, 3)
        metrics["valid"] = invalid_reason is None
        if invalid_reason:
            metrics["invalid_reason"] = invalid_reason
        metrics["score"] = round(score.score(metrics, model_name, stage=stage), 4)
    else:
        mask = (freq_eval >= fmin) & (freq_eval <= fmax)
        invalid_reason = _band_s11_invalid(s11_db, mask)
        band_rl = rl_db[mask]
        finite = np.isfinite(band_rl)
        rl_min = float(np.min(band_rl[finite])) if np.any(finite) else None

        f_target = f0
        pen_f = None
        if f_peak is not None:
            pen_f = ((f_peak - f_target) / f_target) ** 2

        base_metrics = {
            "model": model_name,
            "rl_min_in_band_db": _maybe_round(rl_min, 3),
            "rl_min_db": _maybe_round(rl_min, 3),
            "f_peak_hz": _maybe_round(f_peak, 1),
            "rl_peak_db": _maybe_round(rl_peak, 3),
            "pen_f": _maybe_round(pen_f, 6),
            "constraint_penalty": round(penalty, 3),
        }
        if s11_matched_db_plot is not None and match_meta:
            rl_matched = -s11_matched_db_plot
            band_mask = (freq_plot >= fmin) & (freq_plot <= fmax)
            if np.any(band_mask):
                rl_band = rl_matched[band_mask]
                freq_band = freq_plot[band_mask]
                peak_idx = int(np.argmax(rl_band))
                base_metrics["rl_min_in_band_matched_db"] = _maybe_round(float(np.min(rl_band)), 3)
                base_metrics["rl_peak_in_band_matched_db"] = _maybe_round(float(rl_band[peak_idx]), 3)
                base_metrics["f_peak_in_band_matched_hz"] = _maybe_round(float(freq_band[peak_idx]), 1)
            bw_hz, bw_frac = matching.matched_bandwidth(freq_plot, rl_matched, fmin, fmax, rl_target=10.0)
            series_val = float(match_meta["series_value"])
            shunt_val = float(match_meta["shunt_value"])
            base_metrics.update(
                {
                    "match_topology": match_meta["topology"],
                    "match_series_type": match_meta["series_type"],
                    "match_series_value": series_val,
                    "match_shunt_type": match_meta["shunt_type"],
                    "match_shunt_value": shunt_val,
                    "match_series_value_nh": series_val * 1e9 if match_meta["series_type"] == "L" else float("nan"),
                    "match_series_value_pf": series_val * 1e12 if match_meta["series_type"] == "C" else float("nan"),
                    "match_shunt_value_nh": shunt_val * 1e9 if match_meta["shunt_type"] == "L" else float("nan"),
                    "match_shunt_value_pf": shunt_val * 1e12 if match_meta["shunt_type"] == "C" else float("nan"),
                    "match_penalty": float(match_meta["penalty"]),
                    "match_bandwidth_hz": bw_hz,
                    "match_bandwidth_frac": bw_frac,
                }
            )
        if band_key == "24":
            base_metrics["rl_min_24_db"] = base_metrics["rl_min_in_band_db"]
        else:
            base_metrics["rl_min_5_db"] = base_metrics["rl_min_in_band_db"]

        if invalid_reason:
            base_metrics["valid"] = False
            base_metrics["invalid_reason"] = invalid_reason
            base_metrics["score"] = round(score.score(base_metrics, model_name, stage=stage), 4)
            post.save_metrics(os.path.join(sim_path, "metrics.json"), base_metrics)
            post.save_summary(os.path.join(sim_path, "summary.txt"), base_metrics)
            return base_metrics

        if stage == "lock":
            base_metrics["valid"] = True
            base_metrics["score"] = round(score.score(base_metrics, model_name, stage=stage), 4)
            post.save_metrics(os.path.join(sim_path, "metrics.json"), base_metrics)
            post.save_summary(os.path.join(sim_path, "summary.txt"), base_metrics)
            return base_metrics

        if rl_min is None or rl_min < 8.0:
            base_metrics["valid"] = False
            base_metrics["invalid_reason"] = "rl_min_in_band_below_gate"
            base_metrics["score"] = round(score.score(base_metrics, model_name, stage=stage), 4)
            post.save_metrics(os.path.join(sim_path, "metrics.json"), base_metrics)
            post.save_summary(os.path.join(sim_path, "summary.txt"), base_metrics)
            return base_metrics

        res = nf2ff.CalcNF2FF(
            sim_path,
            f0,
            [90.0],
            [0.0, 180.0],
            radius=1,
            center=[0, 0, 0],
            outfile=f"nf2ff_{band_key}.h5",
            read_cached=False,
        )
        gain_fwd = post.calc_directivity_db(res, 0, 0)
        gain_back = post.calc_directivity_db(res, 0, 1)
        fb = gain_fwd - gain_back
        p_inc = float(post.port_power_inc_at(ports[0], freq_eval, f0))
        p_ref = float(post.port_power_ref_at(ports[0], freq_eval, f0))
        p_acc = float(post.port_power_acc_at(ports[0], freq_eval, f0))
        prad = float(res.Prad[0])
        scale = 1.0
        if math.isfinite(p_inc) and p_inc > 0:
            scale = 1.0 / p_inc
        p_inc_norm = p_inc * scale
        p_ref_norm = p_ref * scale
        p_acc_norm = p_acc * scale
        prad_norm = prad * scale
        u = float(res.P_rad[0][0, 0]) * scale
        p_acc_min = 1e-3
        valid_pacc = math.isfinite(p_acc_norm) and p_acc_norm > p_acc_min
        if valid_pacc:
            realized = 4.0 * math.pi * u / p_acc_norm
            gain_fwd_realized = 10.0 * math.log10(max(realized, 1e-12))
        else:
            gain_fwd_realized = float("nan")
        eff_tol = 1.05
        eff_ok = valid_pacc and math.isfinite(prad_norm) and prad_norm >= 0 and prad_norm <= p_acc_norm * eff_tol
        rad_eff = prad_norm / p_acc_norm if eff_ok and p_acc_norm > 0 else float("nan")
        lam = 299792458.0 / f0
        gain_lin = 10.0 ** (gain_fwd_realized / 10.0) if math.isfinite(gain_fwd_realized) else float("nan")
        eff_ap = (lam * lam / (4.0 * math.pi)) * gain_lin if math.isfinite(gain_lin) else float("nan")

        phi, gains = post.calc_plane_pattern(
            nf2ff,
            sim_path,
            f0,
            range(-180, 181, 5),
            outfile=f"nf2ff_plane_{band_key}.h5",
        )
        post.save_pattern_csv(os.path.join(sim_path, f"pattern_{band_key}.csv"), phi, gains)

        metrics = dict(base_metrics)
        metrics.update(
            {
                "gain_fwd_db": round(gain_fwd, 3),
                "fb_db": round(fb, 3),
                "gain_fwd_realized_db": round(gain_fwd_realized, 3),
                "prad_w": float(prad_norm) if math.isfinite(prad_norm) else float("nan"),
                "p_inc_w": float(p_inc_norm) if math.isfinite(p_inc_norm) else float("nan"),
                "p_ref_w": float(p_ref_norm) if math.isfinite(p_ref_norm) else float("nan"),
                "p_acc_w": float(p_acc_norm) if math.isfinite(p_acc_norm) else float("nan"),
                "p_acc_sign": float(math.copysign(1.0, p_acc_norm)) if math.isfinite(p_acc_norm) and p_acc_norm != 0 else float("nan"),
                "p_inc_w_raw": float(p_inc) if math.isfinite(p_inc) else float("nan"),
                "p_ref_w_raw": float(p_ref) if math.isfinite(p_ref) else float("nan"),
                "p_acc_w_raw": float(p_acc) if math.isfinite(p_acc) else float("nan"),
                "rad_eff": round(rad_eff, 6) if math.isfinite(rad_eff) else float("nan"),
                "rad_eff_pct": round(rad_eff * 100.0, 3) if math.isfinite(rad_eff) else float("nan"),
                "effective_aperture_m2": round(eff_ap, 8) if math.isfinite(eff_ap) else float("nan"),
                "valid": True,
            }
        )
        if band_key == "24":
            metrics["gain_fwd_24_dbi"] = round(gain_fwd, 3)
            metrics["fb_24_db"] = round(fb, 3)
            metrics["gain_fwd_realized_24_dbi"] = round(gain_fwd_realized, 3)
        else:
            metrics["gain_fwd_5_dbi"] = round(gain_fwd, 3)
            metrics["fb_5_db"] = round(fb, 3)
            metrics["gain_fwd_realized_5_dbi"] = round(gain_fwd_realized, 3)
        metrics["score"] = round(score.score(metrics, model_name, stage=stage), 4)

    post.save_metrics(os.path.join(sim_path, "metrics.json"), metrics)
    post.save_summary(os.path.join(sim_path, "summary.txt"), metrics)

    meta_path = os.path.join(sim_path, "meta.json")
    mc.save_json(meta_path, {"nf2ff": meta, "params": params})

    return metrics


def evaluate(
    model_name: str,
    params: Dict,
    substrate: mc.SubstrateConfig,
    constraints: mc.SliceConstraints,
    quality: str,
    run_root: str,
    force: bool = False,
    neighbor: bool = False,
    stage: str = "final",
) -> Dict:
    model = MODELS[model_name]
    merged_params = _merge_params(model.default_params(constraints), params)
    run_dir = compute_run_dir(
        model_name,
        merged_params,
        substrate,
        constraints,
        quality,
        neighbor,
        run_root,
    )
    metrics_path = os.path.join(run_dir, "metrics.json")

    if neighbor:
        if os.path.exists(metrics_path) and not force:
            metrics = mc.load_json(metrics_path)
        else:
            metrics = _run_neighbor(model_name, merged_params, substrate, constraints, quality, run_dir, force)
        return metrics

    if os.path.exists(metrics_path) and not force:
        metrics = mc.load_json(metrics_path)
    else:
        metrics = _run_single(
            model_name,
            merged_params,
            substrate,
            constraints,
            quality,
            run_dir,
            force,
            stage,
        )
    metrics["score"] = round(score.score(metrics, model_name, stage=stage), 4)
    metrics["score_stage"] = stage
    post.save_metrics(metrics_path, metrics)
    post.save_summary(os.path.join(run_dir, "summary.txt"), metrics)
    print_candidate_summary(model_name, metrics)
    return metrics


def compute_run_dir(
    model_name: str,
    params: Dict,
    substrate: mc.SubstrateConfig,
    constraints: mc.SliceConstraints,
    quality: str,
    neighbor: bool,
    run_root: str,
) -> str:
    sparam_version = "sparams_v14"
    payload = {
        "model": model_name,
        "params": params,
        "quality": quality,
        "neighbor": neighbor,
        "constraints": constraints.__dict__,
        "substrate": substrate.__dict__,
        "sparam_version": sparam_version,
    }
    run_hash = mc.hash_params(payload)
    return os.path.join(run_root, run_hash)


def _run_neighbor(
    model_name: str,
    params: Dict,
    substrate: mc.SubstrateConfig,
    constraints: mc.SliceConstraints,
    quality_name: str,
    run_dir: str,
    force: bool,
) -> Dict:
    model = MODELS[model_name]
    primary_band = _primary_band_key(model)
    geom, constraint_info = model.build_geometry(params, constraints)
    penalty = _constraint_penalty(constraints, constraint_info, geom)

    cluster = _build_cluster_geometry(geom, constraints)

    band = list(model.BANDS.values())[0]
    fmin = band["f_low"]
    fmax = band["f_high"]
    f0 = band["f0"]

    quality = mc.quality_from_name(quality_name, fmax, substrate.eps_r, constraints.min_trace_mm)
    fc = _excite_fc(f0, fmin, fmax)

    os.makedirs(run_dir, exist_ok=True)
    mc.save_json(os.path.join(run_dir, "params.json"), {"params": params, "substrate": substrate.__dict__})

    sparams_runs = []
    for excite_idx in range(3):
        port_dir = os.path.join(run_dir, f"port{excite_idx}")
        os.makedirs(port_dir, exist_ok=True)
        FDTD, ports, nf2ff, meta = mc.build_simulation(
            cluster,
            substrate,
            constraints,
            quality,
            f0_hz=f0,
            fc_hz=fc,
            fmax_hz=fmax,
            excite_port=excite_idx,
            port_count=3,
        )
        if force or not os.path.exists(os.path.join(port_dir, "openEMS_log.txt")):
            FDTD.Run(port_dir, cleanup=force, verbose=0)
        freq = np.linspace(fmin, fmax, 201)
        sparams = post.calc_sparams(ports, port_dir, freq, ref_impedance=50, excite_port=excite_idx)
        sparams_runs.append((freq, sparams))

    # isolation from center slice (port 0)
    freq, sparams = sparams_runs[0]
    idx = int(np.argmin(np.abs(freq - f0)))
    s21 = sparams[1][idx]
    s31 = sparams[2][idx]
    iso_db = 20.0 * np.log10(max(abs(s21), abs(s31)) + 1e-12)

    metrics = {
        "model": model_name,
        "iso_db": round(float(iso_db), 3),
        "constraint_penalty": round(penalty, 3),
        "valid": True,
    }
    if primary_band == "24":
        metrics["iso_24_db"] = metrics["iso_db"]
    elif primary_band == "5":
        metrics["iso_5_db"] = metrics["iso_db"]

    post.save_metrics(os.path.join(run_dir, "metrics.json"), metrics)
    post.save_summary(os.path.join(run_dir, "summary.txt"), metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default=None)
    parser.add_argument("--params", default=None, help="Path to params JSON")
    parser.add_argument("--quality", default="fast", choices=["fast", "medium", "high"])
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--neighbor", action="store_true")
    parser.add_argument("--stage", choices=["lock", "final"], default="final")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_root = args.run_root or os.path.join(root, "runs")
    constraints_path = os.path.join(root, "designs", "constraints.json")

    try:
        mc.require_openems()
    except mc.OpenEMSImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = _load_params(args.params)
    model_name = args.model or payload.get("model") or "dir24"
    model = MODELS[model_name]
    constraints = mc.load_constraints(constraints_path)
    constraints = mc.apply_constraint_overrides(
        constraints, getattr(model, "DEFAULT_CONSTRAINTS", None)
    )
    resolved = _resolve_payload(model_name, payload, constraints)

    metrics = evaluate(
        resolved["model_name"],
        resolved["params"],
        resolved["substrate"],
        constraints,
        args.quality,
        run_root,
        force=args.force,
        neighbor=args.neighbor,
        stage=args.stage,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

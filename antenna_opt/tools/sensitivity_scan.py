"""Parameter sensitivity scan for resonance and peak return loss."""

from __future__ import annotations

import argparse
import math
import os

from sim import model_common as mc
from sim import run_one
from sim import model_dir24_quasiyagi
from sim import model_dir24_acs_monopole_yagi
from sim import model_dir24_acs_ursi_uniplanar
from sim import model_dir24_acs_ursi_ground
from sim import model_dir5_quasiyagi
from sim import model_dir5_ms_cps_dipole_yagi
from sim import model_omni_dual_ifa


MODEL_MAP = {
    "dir24": model_dir24_quasiyagi,
    "dir24_acs": model_dir24_acs_monopole_yagi,
    "dir24_acs_ursi_uni": model_dir24_acs_ursi_uniplanar,
    "dir24_acs_ursi_gnd": model_dir24_acs_ursi_ground,
    "dir5": model_dir5_quasiyagi,
    "dir5_ms_cps": model_dir5_ms_cps_dipole_yagi,
    "omni": model_omni_dual_ifa,
}


def _load_payload(path: str | None) -> dict:
    if not path:
        return {}
    return mc.load_json(path)


def _maybe_delta(val: float | None, base: float | None, scale: float) -> float:
    if val is None or base is None:
        return float("nan")
    if not (math.isfinite(val) and math.isfinite(base)):
        return float("nan")
    return (val - base) * scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_MAP.keys()), required=True)
    parser.add_argument("--params", default=None, help="Path to params JSON")
    parser.add_argument("--delta", type=float, default=0.1, help="Fractional perturbation")
    parser.add_argument("--quality", choices=["fast", "medium", "high"], default="fast")
    parser.add_argument("--stage", choices=["lock", "final"], default="lock")
    parser.add_argument("--run-root", default=None)
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    constraints = mc.load_constraints(os.path.join(root, "designs", "constraints.json"))
    model = MODEL_MAP[args.model]
    constraints = mc.apply_constraint_overrides(
        constraints, getattr(model, "DEFAULT_CONSTRAINTS", None)
    )
    base_params = model.default_params(constraints)
    payload = _load_payload(args.params)
    overrides = payload.get("params", payload)
    base_params.update(overrides)
    substrate = mc.load_substrate(payload.get("substrate"), base=getattr(model, "DEFAULT_SUBSTRATE", None))

    run_root = args.run_root or os.path.join(root, "runs")
    baseline = run_one.evaluate(
        args.model,
        base_params,
        substrate,
        constraints,
        args.quality,
        run_root,
        force=False,
        stage=args.stage,
    )

    base_f = baseline.get("f_peak_hz")
    base_rl = baseline.get("rl_peak_db")
    if base_f is not None:
        print(f"Baseline f_peak={base_f/1e9:.3f} GHz RL_peak={base_rl:.2f} dB")
    else:
        print("Baseline f_peak=nan RL_peak=nan")

    bounds = model.param_bounds()
    header = "param,df_peak_minus_mhz,dRL_peak_minus_db,df_peak_plus_mhz,dRL_peak_plus_db"
    print(header)

    for key in bounds:
        low, high = bounds[key]
        base_val = float(base_params[key])
        if base_val == 0.0:
            shift = (high - low) * args.delta
            val_minus = max(base_val - shift, low)
            val_plus = min(base_val + shift, high)
        else:
            val_minus = max(base_val * (1.0 - args.delta), low)
            val_plus = min(base_val * (1.0 + args.delta), high)

        params_minus = dict(base_params)
        params_plus = dict(base_params)
        params_minus[key] = val_minus
        params_plus[key] = val_plus

        met_minus = run_one.evaluate(
            args.model,
            params_minus,
            substrate,
            constraints,
            args.quality,
            run_root,
            force=False,
            stage=args.stage,
        )
        met_plus = run_one.evaluate(
            args.model,
            params_plus,
            substrate,
            constraints,
            args.quality,
            run_root,
            force=False,
            stage=args.stage,
        )

        df_minus = _maybe_delta(met_minus.get("f_peak_hz"), base_f, 1e-6)
        df_plus = _maybe_delta(met_plus.get("f_peak_hz"), base_f, 1e-6)
        d_rl_minus = _maybe_delta(met_minus.get("rl_peak_db"), base_rl, 1.0)
        d_rl_plus = _maybe_delta(met_plus.get("rl_peak_db"), base_rl, 1.0)

        print(f"{key},{df_minus:.3f},{d_rl_minus:.3f},{df_plus:.3f},{d_rl_plus:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

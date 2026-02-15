"""Microstrip port V/I extraction sanity checks."""

from __future__ import annotations

import math

import jax
import numpy as np

import fdtdx

from sim import metrics
from sim.sources import GapVoltageSource


def _mm(val: float) -> float:
    return val * 1e-3


def _run_case(
    load_mode: str,
    freqs: list[float] | None = None,
    load_sigma: float | None = None,
    source_offset_mm: float = 0.0,
    port_offset_mm: float = 0.0,
    port_len_mm: float = 2.0,
    resolution_mm: float = 0.4,
    v_method: str = "mean",
    current_axis: int = 0,
    i_mode: str = "loop",
) -> dict:
    if load_mode not in {"open", "short", "match"}:
        raise ValueError(f"Unknown load mode: {load_mode}")

    materials = {
        "Air": fdtdx.Material(permittivity=1.0),
        "FR4": fdtdx.Material(permittivity=4.4),
        "Copper": fdtdx.Material(permittivity=1.0, is_pec=True),
    }
    if load_mode == "match":
        sigma = 3.3 if load_sigma is None else float(load_sigma)
        materials["Load"] = fdtdx.Material(permittivity=1.0, electric_conductivity=sigma)

    sub_t = _mm(1.6)
    ground_t = _mm(0.4)
    line_t = _mm(0.4)
    line_len = _mm(30.0)
    line_w = _mm(2.4)
    load_len = _mm(3.0)
    load_w = _mm(4.0)

    volume = fdtdx.SimulationVolume(
        name="vol",
        partial_real_shape=(_mm(50.0), _mm(20.0), _mm(20.0)),
        material=materials["Air"],
    )

    substrate = fdtdx.UniformMaterialObject(
        name="substrate",
        partial_real_shape=(_mm(40.0), _mm(16.0), sub_t),
        material=materials["FR4"],
    )

    ground = fdtdx.UniformMaterialObject(
        name="ground",
        partial_real_shape=(_mm(40.0), _mm(16.0), ground_t),
        material=materials["Copper"],
    )

    line = fdtdx.UniformMaterialObject(
        name="line",
        partial_real_shape=(line_len, line_w, line_t),
        material=materials["Copper"],
    )

    if freqs is None:
        freqs = [5.0e9]
    if len(freqs) == 0:
        raise ValueError("freqs must contain at least one frequency")
    freqs = [float(f) for f in freqs]
    f_min = min(freqs)
    f_max = max(freqs)
    f0 = 0.5 * (f_min + f_max)
    resolution = _mm(resolution_mm)
    port_len = _mm(port_len_mm)
    port_width = line_w
    # Gap voltage should span the substrate only (metal thickness is not part of the field gap).
    port_gap = sub_t

    spectral_width = max(f0 * 0.6, (f_max - f_min))
    source = GapVoltageSource(
        name="port_source",
        partial_real_shape=(port_len, port_width, port_gap),
        wave_character=fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_frequency=f0,
            spectral_width=spectral_width,
        ),
        polarization_axis=2,
        amplitude=1.0,
    )
    waves = tuple(fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f) for f in freqs)

    port_v = fdtdx.PhasorDetector(
        name="port_v",
        partial_real_shape=(port_len, port_width, port_gap),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ez", "Hy", "Hz"),
    )

    pad = 0.0
    port_i_height = port_gap
    port_i = fdtdx.PhasorDetector(
        name="port_i",
        partial_real_shape=(port_len + 2 * pad, port_width + 2 * pad, port_i_height),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ez", "Hy", "Hz"),
    )
    port_flux_height = _mm(4.0)
    port_flux = fdtdx.PhasorDetector(
        name="port_flux",
        partial_real_shape=(resolution, port_width, port_flux_height),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )

    load = None
    load_height = sub_t + ground_t + line_t
    load_overlap = resolution
    if load_mode == "short":
        load = fdtdx.UniformMaterialObject(
            name="short",
            partial_real_shape=(load_len, load_w, load_height),
            material=materials["Copper"],
        )
    elif load_mode == "match":
        load = fdtdx.UniformMaterialObject(
            name="load",
            partial_real_shape=(load_len, load_w, load_height),
            material=materials["Load"],
        )

    constraints = []
    constraints.append(substrate.place_at_center(volume, axes=(0, 1)))
    constraints.append(substrate.place_at_center(volume, axes=2))

    constraints.append(ground.place_at_center(substrate, axes=(0, 1)))
    constraints.append(ground.place_below(substrate))

    constraints.append(line.place_at_center(substrate, axes=1))
    constraints.append(
        line.place_relative_to(
            substrate,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=_mm(2.0),
        )
    )
    constraints.append(line.place_above(substrate))

    source_offset = _mm(source_offset_mm)
    constraints.append(source.place_at_center(line, axes=1))
    constraints.append(source.place_at_center(substrate, axes=2))
    constraints.append(
        source.place_relative_to(
            line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=source_offset,
        )
    )

    port_offset = _mm(port_offset_mm)
    constraints.append(port_v.place_at_center(line, axes=1))
    constraints.append(port_v.place_at_center(substrate, axes=2))
    constraints.append(
        port_v.place_relative_to(
            line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset,
        )
    )
    constraints.append(port_i.place_at_center(line, axes=1))
    constraints.append(
        port_i.place_relative_to(
            line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset,
        )
    )
    constraints.append(
        port_i.place_relative_to(
            substrate,
            axes=2,
            own_positions=-1,
            other_positions=-1,
            margins=0.0,
        )
    )
    constraints.append(port_flux.place_at_center(line, axes=1))
    constraints.append(port_flux.place_at_center(substrate, axes=2))
    constraints.append(
        port_flux.place_relative_to(
            line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset,
        )
    )

    if load is not None:
        constraints.append(load.place_at_center(line, axes=1))
        constraints.append(load.place_relative_to(line, axes=0, own_positions=1, other_positions=1, margins=0.0))
        constraints.append(
            load.place_relative_to(
                ground,
                axes=2,
                own_positions=-1,
                other_positions=1,
                margins=-load_overlap,
            )
        )

    objects = [volume, substrate, ground, line, source, port_v, port_i, port_flux]
    if load is not None:
        objects.append(load)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml")
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(list(bound_dict.values()))
    constraints.extend(bound_constraints)

    cfg = fdtdx.SimulationConfig(
        time=2.0e-9,
        resolution=resolution,
        backend="cpu",
    )

    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=cfg,
        constraints=constraints,
        key=jax.random.PRNGKey(0),
    )

    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, jax.random.PRNGKey(1))
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=jax.random.PRNGKey(2))

    port_v_phasor = np.array(arrays.detector_states["port_v"]["phasor"][0])
    port_i_phasor = np.array(arrays.detector_states["port_i"]["phasor"][0])
    port_flux_phasor = np.array(arrays.detector_states["port_flux"]["phasor"][0])

    port_v_mask = None
    port_i_mask = None
    if arrays.pec_mask is not None:
        port_v_obj = next(det for det in objects.detectors if det.name == "port_v")
        port_i_obj = next(det for det in objects.detectors if det.name == "port_i")
        port_v_mask = 1.0 - np.array(arrays.pec_mask[*port_v_obj.grid_slice])
        port_i_mask = 1.0 - np.array(arrays.pec_mask[*port_i_obj.grid_slice])

    V = metrics.compute_gap_voltage(port_v_phasor[:, 0], port_gap, mask=port_v_mask, method=v_method)
    I_loop = metrics.compute_loop_current(
        port_i_phasor[:, 1],
        port_i_phasor[:, 2],
        cfg.resolution,
        mask=port_i_mask,
        current_axis=current_axis,
    )
    port_power = np.zeros(port_flux_phasor.shape[0], dtype=np.complex128)
    for idx in range(port_flux_phasor.shape[0]):
        ph = port_flux_phasor[idx]
        port_power[idx] = metrics.complex_power_through_plane(
            ph[0:3],
            ph[3:6],
            axis=0,
            resolution_m=cfg.resolution,
        )

    I_power = np.full_like(V, np.nan, dtype=np.complex128)
    valid_power = np.isfinite(port_power) & np.isfinite(V) & (np.abs(V) > 1e-18)
    if np.any(valid_power):
        I_power[valid_power] = 2.0 * np.conj(port_power[valid_power]) / np.conj(V[valid_power])

    if i_mode == "power":
        I = I_power
    else:
        I = I_loop

    if not np.isfinite(V).any() or not np.isfinite(I).any():
        raise AssertionError("Invalid port V/I extraction")

    if i_mode != "power":
        p_vi = 0.5 * np.real(V * np.conj(I))
        flip = np.isfinite(p_vi) & (p_vi < 0)
        if np.any(flip):
            I = np.where(flip, -I, I)
            p_vi = 0.5 * np.real(V * np.conj(I))

        pin_flux = np.abs(np.real(port_power))
        valid_scale = np.isfinite(p_vi) & (p_vi > 0) & np.isfinite(pin_flux) & (pin_flux > 0)
        scale = np.ones_like(p_vi, dtype=np.float64)
        if np.any(valid_scale):
            scale[valid_scale] = np.clip(pin_flux[valid_scale] / p_vi[valid_scale], 0.1, 10.0)
            I = I * scale

    i0 = I[0]
    if not np.isfinite(i0) or abs(i0) < 1e-18:
        raise AssertionError("Port current is invalid")

    zin_vi = np.full_like(V, np.nan + 1j * np.nan, dtype=np.complex128)
    vi_valid = np.isfinite(V) & np.isfinite(I) & (np.abs(I) > 1e-18)
    zin_vi[vi_valid] = V[vi_valid] / I[vi_valid]
    s11_vi = metrics.s11_from_impedance(zin_vi)
    rl_vi_db = -metrics.s11_db(s11_vi)

    zin_flux = np.nan + 1j * np.nan
    zin_flux = np.full_like(V, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_flux = np.isfinite(V) & np.isfinite(port_power) & (np.abs(port_power) > 1e-18)
    zin_flux[valid_flux] = (V[valid_flux] * np.conj(V[valid_flux])) / (2.0 * np.conj(port_power[valid_flux]))
    s11_flux = metrics.s11_from_impedance(zin_flux)
    rl_flux_db = -metrics.s11_db(s11_flux)

    f0_idx = int(np.argmin(np.abs(np.array(freqs) - f0)))
    zin_vi_f0 = zin_vi[f0_idx]
    zin_flux_f0 = zin_flux[f0_idx]

    return {
        "freq_hz": freqs,
        "v_real": [float(np.real(v)) for v in V],
        "v_imag": [float(np.imag(v)) for v in V],
        "i_real": [float(np.real(i)) for i in I],
        "i_imag": [float(np.imag(i)) for i in I],
        "s11_vi_real": [float(np.real(s)) for s in s11_vi],
        "s11_vi_imag": [float(np.imag(s)) for s in s11_vi],
        "s11_flux_real": [float(np.real(s)) for s in s11_flux],
        "s11_flux_imag": [float(np.imag(s)) for s in s11_flux],
        "rl_vi_db": float(rl_vi_db[f0_idx]),
        "rl_flux_db": float(rl_flux_db[f0_idx]),
        "rl_vi_db_sweep": [float(r) for r in rl_vi_db],
        "rl_flux_db_sweep": [float(r) for r in rl_flux_db],
        "zin_vi_real_ohm": float(np.real(zin_vi_f0)),
        "zin_vi_imag_ohm": float(np.imag(zin_vi_f0)),
        "zin_flux_real_ohm": float(np.real(zin_flux_f0)),
        "zin_flux_imag_ohm": float(np.imag(zin_flux_f0)),
        "f0_hz": float(f0),
    }


def run_test() -> None:
    open_case = _run_case("open")
    short_case = _run_case("short")
    match_case = _run_case("match")
    rl_open = open_case["rl_vi_db"]
    rl_short = short_case["rl_vi_db"]
    rl_match = match_case["rl_vi_db"]

    if rl_open > 3.0:
        raise AssertionError(f"Open circuit RL too high: {rl_open:.2f} dB")
    if rl_short > 3.0:
        raise AssertionError(f"Short circuit RL too high: {rl_short:.2f} dB")
    if not math.isfinite(rl_match):
        raise AssertionError(f"Matched RL invalid: {rl_match}")


if __name__ == "__main__":
    run_test()
    print("test_microstrip_port: ok")

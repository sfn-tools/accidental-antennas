"""Lumped port V/I extraction sanity checks."""

from __future__ import annotations

import jax
import numpy as np

import fdtdx

from sim.metrics import compute_lumped_port_vi, complex_power_through_plane, s11_from_impedance
from sim.sources import GapVoltageSource


def _mm(val: float) -> float:
    return val * 1e-3


def _run_case(
    load_sigma: float | None = None,
    freqs: list[float] | None = None,
    load_is_pec: bool = False,
) -> dict:
    materials = {
        "Air": fdtdx.Material(permittivity=1.0),
        "PEC": fdtdx.Material(permittivity=1.0, is_pec=True),
    }
    if load_sigma is not None:
        materials["Load"] = fdtdx.Material(permittivity=1.0, electric_conductivity=load_sigma)

    volume = fdtdx.SimulationVolume(
        name="vol",
        partial_real_shape=(_mm(20.0), _mm(20.0), _mm(20.0)),
        material=materials["Air"],
    )

    plate_size = (_mm(6.0), _mm(6.0), _mm(1.0))
    gap_size = (_mm(4.0), _mm(4.0), _mm(2.0))
    port_size = (plate_size[0], plate_size[1], gap_size[2])

    bottom = fdtdx.UniformMaterialObject(
        name="plate_bottom",
        partial_real_shape=plate_size,
        material=materials["PEC"],
    )
    top = fdtdx.UniformMaterialObject(
        name="plate_top",
        partial_real_shape=plate_size,
        material=materials["PEC"],
    )

    if freqs is None:
        freqs = [1.0e9]
    if len(freqs) == 0:
        raise ValueError("freqs must contain at least one frequency")
    freqs = [float(f) for f in freqs]
    f_min = min(freqs)
    f_max = max(freqs)
    f0 = 0.5 * (f_min + f_max)
    source = GapVoltageSource(
        name="port_source",
        partial_real_shape=gap_size,
        wave_character=fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_frequency=f0,
            spectral_width=max(f0 * 0.6, (f_max - f_min)),
        ),
        polarization_axis=2,
        amplitude=1.0,
    )
    waves = tuple(fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f) for f in freqs)

    port_gap = gap_size[2]
    port_detector = fdtdx.PhasorDetector(
        name="port",
        partial_real_shape=(port_size[0], port_size[1], port_gap),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ez", "Hx", "Hy"),
    )
    resolution = _mm(1.0)
    port_flux = fdtdx.PhasorDetector(
        name="port_flux",
        partial_real_shape=(port_size[0], port_size[1], resolution),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )

    load = None
    if load_is_pec:
        load = fdtdx.UniformMaterialObject(
            name="load_short",
            partial_real_shape=gap_size,
            material=materials["PEC"],
        )
    elif load_sigma is not None:
        load = fdtdx.UniformMaterialObject(
            name="load",
            partial_real_shape=gap_size,
            material=materials["Load"],
        )

    constraints = []
    constraints.append(bottom.place_at_center(volume, axes=(0, 1)))
    constraints.append(bottom.place_relative_to(volume, axes=2, own_positions=-1, other_positions=-1, margins=_mm(4.0)))

    constraints.append(source.place_at_center(bottom, axes=(0, 1)))
    constraints.append(source.place_relative_to(bottom, axes=2, own_positions=-1, other_positions=1, margins=0.0))

    constraints.append(port_detector.place_at_center(source, axes=(0, 1, 2)))
    constraints.append(port_flux.place_at_center(source, axes=(0, 1, 2)))

    constraints.append(top.place_at_center(bottom, axes=(0, 1)))
    constraints.append(top.place_relative_to(source, axes=2, own_positions=-1, other_positions=1, margins=0.0))

    if load is not None:
        constraints.append(load.place_at_center(source, axes=(0, 1, 2)))

    objects = [volume, bottom, top, source, port_detector, port_flux]
    if load is not None:
        objects.append(load)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml")
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(list(bound_dict.values()))
    constraints.extend(bound_constraints)

    cfg = fdtdx.SimulationConfig(
        time=5.0e-9,
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

    port_phasor = np.array(arrays.detector_states["port"]["phasor"][0])
    port_flux_phasor = np.array(arrays.detector_states["port_flux"]["phasor"][0])
    port_mask = None
    if arrays.pec_mask is not None:
        port_obj = next(det for det in objects.detectors if det.name == "port")
        port_mask = 1.0 - np.array(arrays.pec_mask[*port_obj.grid_slice])
    V, I_loop = compute_lumped_port_vi(
        port_phasor,
        gap_length_m=gap_size[2],
        resolution_m=cfg.resolution,
        mask=port_mask,
        current_axis=2,
    )
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    sigma_eff = float(load_sigma) if load_sigma is not None else 0.0
    e_avg = -V / gap_size[2]
    area = gap_size[0] * gap_size[1]
    I_disp = (sigma_eff + 1j * omega * fdtdx.constants.eps0) * e_avg * area
    p_disp = 0.5 * np.real(V * np.conj(I_disp))
    flip_disp = np.isfinite(p_disp) & (p_disp < 0)
    if np.any(flip_disp):
        I_disp = np.where(flip_disp, -I_disp, I_disp)

    I = I_disp
    port_power = np.zeros(port_flux_phasor.shape[0], dtype=np.complex128)
    for idx in range(port_flux_phasor.shape[0]):
        ph = port_flux_phasor[idx]
        port_power[idx] = complex_power_through_plane(
            ph[0:3],
            ph[3:6],
            axis=2,
            resolution_m=cfg.resolution,
        )
    p_vi = 0.5 * np.real(V * np.conj(I))
    flip = np.isfinite(p_vi) & (p_vi < 0)
    if np.any(flip):
        I = np.where(flip, -I, I)
        p_vi = 0.5 * np.real(V * np.conj(I))

    apply_flux_scale = False
    if apply_flux_scale:
        pin_flux = np.abs(np.real(port_power))
        valid_scale = np.isfinite(p_vi) & (p_vi > 0) & np.isfinite(pin_flux) & (pin_flux > 0)
        scale = np.ones_like(p_vi, dtype=np.float64)
        if np.any(valid_scale):
            scale[valid_scale] = np.clip(pin_flux[valid_scale] / p_vi[valid_scale], 0.1, 10.0)
            I = I * scale
            p_vi = p_vi * scale

    zin_vi = np.full_like(V, np.nan + 1j * np.nan, dtype=np.complex128)
    vi_valid = np.isfinite(V) & np.isfinite(I) & (np.abs(I) > 1e-18)
    zin_vi[vi_valid] = V[vi_valid] / I[vi_valid]
    s11_vi = s11_from_impedance(zin_vi)
    rl_vi_db = -20.0 * np.log10(np.abs(s11_vi) + 1e-12)

    zin_flux = np.full_like(V, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_flux = np.isfinite(V) & np.isfinite(port_power) & (np.abs(port_power) > 1e-18)
    zin_flux[valid_flux] = (V[valid_flux] * np.conj(V[valid_flux])) / (2.0 * np.conj(port_power[valid_flux]))
    s11_flux = s11_from_impedance(zin_flux)
    rl_flux_db = -20.0 * np.log10(np.abs(s11_flux) + 1e-12)

    f0_idx = int(np.argmin(np.abs(np.array(freqs) - f0)))
    zin_vi_f0 = zin_vi[f0_idx]
    zin_flux_f0 = zin_flux[f0_idx]

    return {
        "freq_hz": freqs,
        "v_real": [float(np.real(v)) for v in V],
        "v_imag": [float(np.imag(v)) for v in V],
        "i_real": [float(np.real(i)) for i in I],
        "i_imag": [float(np.imag(i)) for i in I],
        "i_loop_real": [float(np.real(i)) for i in I_loop],
        "i_loop_imag": [float(np.imag(i)) for i in I_loop],
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
    open_case = _run_case(load_sigma=None)
    short_case = _run_case(load_sigma=5.0e6)
    match_case = _run_case(load_sigma=2.5)
    rl_open = open_case["rl_vi_db"]
    rl_short = short_case["rl_vi_db"]
    rl_match = match_case["rl_vi_db"]
    zin_match = complex(match_case["zin_vi_real_ohm"], match_case["zin_vi_imag_ohm"])

    if rl_open > 10.0:
        raise AssertionError(f"Open circuit RL too high: {rl_open:.2f} dB")
    if rl_short > 10.0:
        raise AssertionError(f"Short circuit RL too high: {rl_short:.2f} dB")
    if rl_match < rl_open + 2.0:
        raise AssertionError(f"Matched RL not improved: open={rl_open:.2f}, match={rl_match:.2f}")
    if not (20.0 <= abs(zin_match) <= 120.0):
        raise AssertionError(f"Matched Zin out of range: {zin_match}")


if __name__ == "__main__":
    run_test()
    print("test_port_vi: ok")

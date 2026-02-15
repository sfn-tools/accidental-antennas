"""Basic PEC enforcement test for FDTDX."""

from __future__ import annotations

import math

import jax
import numpy as np

import fdtdx

from sim.sources import GapVoltageSource


def _mm(val: float) -> float:
    return val * 1e-3


def run_test() -> None:
    materials = {
        "Air": fdtdx.Material(permittivity=1.0),
        "PEC": fdtdx.Material(permittivity=1.0, is_pec=True),
    }

    volume = fdtdx.SimulationVolume(
        name="vol",
        partial_real_shape=(_mm(20.0), _mm(20.0), _mm(20.0)),
        material=materials["Air"],
    )

    pec_block = fdtdx.UniformMaterialObject(
        name="pec_block",
        partial_real_shape=(_mm(6.0), _mm(6.0), _mm(6.0)),
        material=materials["PEC"],
    )

    f0 = 2.0e9
    source = GapVoltageSource(
        name="src",
        partial_real_shape=(_mm(3.0), _mm(3.0), _mm(3.0)),
        wave_character=fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_frequency=f0,
            spectral_width=f0 * 0.6,
        ),
        polarization_axis=2,
        amplitude=1.0,
    )

    det_inside = fdtdx.FieldDetector(
        name="det_inside",
        partial_real_shape=(_mm(3.0), _mm(3.0), _mm(3.0)),
        reduce_volume=True,
        components=("Ex", "Ey", "Ez"),
    )

    det_outside = fdtdx.FieldDetector(
        name="det_outside",
        partial_real_shape=(_mm(3.0), _mm(3.0), _mm(3.0)),
        reduce_volume=True,
        components=("Ex", "Ey", "Ez"),
    )

    constraints = []
    constraints.append(pec_block.place_at_center(volume, axes=(0, 1, 2)))
    constraints.append(det_inside.place_at_center(pec_block, axes=(0, 1, 2)))

    constraints.append(source.place_at_center(volume, axes=(1, 2)))
    constraints.append(
        source.place_relative_to(
            volume,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=_mm(2.0),
        )
    )

    constraints.append(det_outside.place_at_center(volume, axes=(1, 2)))
    constraints.append(
        det_outside.place_relative_to(
            volume,
            axes=0,
            own_positions=1,
            other_positions=1,
            margins=-_mm(2.0),
        )
    )

    cfg = fdtdx.SimulationConfig(
        time=0.5e-9,
        resolution=_mm(1.0),
        backend="cpu",
    )

    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, pec_block, source, det_inside, det_outside],
        config=cfg,
        constraints=constraints,
        key=jax.random.PRNGKey(0),
    )

    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, jax.random.PRNGKey(1))
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=jax.random.PRNGKey(2))

    inside = np.array(arrays.detector_states["det_inside"]["fields"])
    outside = np.array(arrays.detector_states["det_outside"]["fields"])
    if inside.size == 0 or outside.size == 0:
        raise AssertionError("Detector states are empty")

    inside_mag = float(np.max(np.abs(inside)))
    outside_mag = float(np.max(np.abs(outside)))
    if not math.isfinite(outside_mag) or outside_mag <= 0:
        raise AssertionError("Outside field magnitude invalid")

    ratio = inside_mag / outside_mag
    if ratio > 1e-2:
        raise AssertionError(f"PEC enforcement too weak: inside/outside={ratio:.3e}")


if __name__ == "__main__":
    run_test()
    print("test_pec: ok")

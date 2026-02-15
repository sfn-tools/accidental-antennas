"""Run one FDTDX antenna simulation and extract metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

from . import calibration, common, geometry, matching, metrics, post
from .params import resample_params
from .models import MODEL_CONFIGS, ModelConfig, adjust_model_for_resolution
from .sources import GapVoltageSource


def _init_device_params(
    shape: tuple[int, int, int],
    mode: str,
    *,
    rng: np.random.Generator | None = None,
    init_blob_density: float = 0.45,
    init_blob_smooth_iters: int = 2,
    init_walk_length_frac: float = 0.7,
    init_walk_length_cells: int = 0,
    init_walk_branches: int = 3,
    init_walk_thickness: int = 2,
    init_walk_turn_prob: float = 0.3,
    init_grid_size: int = 5,
    init_grid_connect_prob: float = 0.5,
    init_grid_thickness: int = 2,
) -> jax.Array:
    if rng is None:
        rng = np.random.default_rng(0)

    def add_feed_spine(arr: np.ndarray, nx: int, ny: int, nz: int) -> None:
        spine_len = max(1, int(round(nx * 0.2)))
        spine_w = max(1, int(round(ny * 0.08)))
        y0 = max(0, (ny // 2) - spine_w)
        y1 = min(ny, (ny // 2) + spine_w)
        arr[:spine_len, y0:y1, :] = 1.0

    def smooth2d(field: np.ndarray, iters: int) -> np.ndarray:
        out = field
        for _ in range(max(0, int(iters))):
            out = (
                out
                + np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
                + np.roll(np.roll(out, 1, axis=0), 1, axis=1)
                + np.roll(np.roll(out, 1, axis=0), -1, axis=1)
                + np.roll(np.roll(out, -1, axis=0), 1, axis=1)
                + np.roll(np.roll(out, -1, axis=0), -1, axis=1)
            ) / 9.0
        return out

    def stamp(arr: np.ndarray, x: int, y: int, thickness: int, nx: int, ny: int) -> None:
        x0 = max(0, x - thickness)
        x1 = min(nx, x + thickness + 1)
        y0 = max(0, y - thickness)
        y1 = min(ny, y + thickness + 1)
        arr[x0:x1, y0:y1, :] = 1.0

    def draw_segment(
        arr: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        thickness: int,
        nx: int,
        ny: int,
    ) -> None:
        if x0 == x1:
            step = 1 if y1 >= y0 else -1
            for y in range(y0, y1 + step, step):
                stamp(arr, x0, y, thickness, nx, ny)
        else:
            step = 1 if x1 >= x0 else -1
            for x in range(x0, x1 + step, step):
                stamp(arr, x, y0, thickness, nx, ny)

    if mode == "solid":
        arr = np.ones(shape, dtype=np.float32)
    elif mode == "empty":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        add_feed_spine(arr, nx, ny, nz)
    elif mode == "uniform":
        arr = np.full(shape, 0.5, dtype=np.float32)
    elif mode == "patch":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        x0 = int(nx * 0.2)
        x1 = int(nx * 0.8)
        y0 = int(ny * 0.35)
        y1 = int(ny * 0.65)
        arr[x0:x1, y0:y1, :] = 1.0
        add_feed_spine(arr, nx, ny, nz)
    elif mode == "seed_blob":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        density = float(np.clip(init_blob_density, 0.0, 1.0))
        noise = rng.random((nx, ny)).astype(np.float32)
        noise = smooth2d(noise, init_blob_smooth_iters)
        if density > 0.0:
            thresh = np.quantile(noise, 1.0 - density)
            mask = noise >= thresh
            arr[mask, :] = 1.0
        add_feed_spine(arr, nx, ny, nz)
    elif mode == "random_walk":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        add_feed_spine(arr, nx, ny, nz)
        steps = int(init_walk_length_cells)
        if steps <= 0:
            steps = int(round(max(float(init_walk_length_frac), 0.05) * nx))
        branches = max(1, int(init_walk_branches))
        thickness = max(0, int(init_walk_thickness))
        turn_prob = float(np.clip(init_walk_turn_prob, 0.0, 1.0))
        start_x = min(max(1, int(round(nx * 0.2))), nx - 1)
        start_y = ny // 2
        dirs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)
        for _ in range(branches):
            x, y = start_x, start_y
            d = int(rng.integers(0, 4))
            steps_b = max(1, steps // branches)
            for _ in range(steps_b):
                if rng.random() < turn_prob:
                    d = int(rng.integers(0, 4))
                dx, dy = dirs[d]
                x = int(np.clip(x + dx, 0, nx - 1))
                y = int(np.clip(y + dy, 0, ny - 1))
                stamp(arr, x, y, thickness, nx, ny)
    elif mode == "stochastic_grid":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        add_feed_spine(arr, nx, ny, nz)
        grid_size = max(3, int(init_grid_size))
        thickness = max(0, int(init_grid_thickness))
        prob = float(np.clip(init_grid_connect_prob, 0.0, 1.0))
        gx = np.linspace(0, nx - 1, grid_size, dtype=int)
        gy = np.linspace(0, ny - 1, grid_size, dtype=int)
        row = grid_size // 2
        for ix in range(grid_size - 1):
            next_row = int(np.clip(row + rng.integers(-1, 2), 0, grid_size - 1))
            draw_segment(arr, gx[ix], gy[row], gx[ix + 1], gy[next_row], thickness, nx, ny)
            row = next_row
        for ix in range(grid_size):
            for iy in range(grid_size):
                if ix + 1 < grid_size and rng.random() < prob:
                    draw_segment(arr, gx[ix], gy[iy], gx[ix + 1], gy[iy], thickness, nx, ny)
                if iy + 1 < grid_size and rng.random() < prob:
                    draw_segment(arr, gx[ix], gy[iy], gx[ix], gy[iy + 1], thickness, nx, ny)
    elif mode == "yagi":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        if nx < 6 or ny < 6:
            arr[:] = 1.0
        else:
            y_margin = max(1, int(round(ny * 0.2)))
            y0 = y_margin
            y1 = max(y0 + 1, ny - y_margin)
            bar_w = max(1, int(round(nx * 0.03)))

            x_driver = max(1, int(round(nx * 0.15)))
            x_dir1 = max(x_driver + bar_w + 1, int(round(nx * 0.55)))
            x_dir2 = max(x_dir1 + bar_w + 1, int(round(nx * 0.8)))

            spine_w = max(1, int(round(ny * 0.08)))
            spine_y0 = max(0, (ny // 2) - spine_w)
            spine_y1 = min(ny, (ny // 2) + spine_w)
            arr[:x_driver, spine_y0:spine_y1, :] = 1.0

            arr[x_driver:x_driver + bar_w, y0:y1, :] = 1.0

            dir1_span = max(1, int(round((y1 - y0) * 0.8)))
            dir1_y0 = max(0, (ny - dir1_span) // 2)
            dir1_y1 = min(ny, dir1_y0 + dir1_span)
            arr[x_dir1:x_dir1 + bar_w, dir1_y0:dir1_y1, :] = 1.0

            dir2_span = max(1, int(round((y1 - y0) * 0.6)))
            dir2_y0 = max(0, (ny - dir2_span) // 2)
            dir2_y1 = min(ny, dir2_y0 + dir2_span)
            arr[x_dir2:x_dir2 + bar_w, dir2_y0:dir2_y1, :] = 1.0
    elif mode == "inverted_f":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        if nx < 4 or ny < 4:
            arr[:] = 1.0
        else:
            spine_x = max(1, int(round(nx * 0.5)))
            spine_w = max(1, int(round(ny * 0.08)))
            y0 = max(0, (ny // 2) - spine_w)
            y1 = min(ny, (ny // 2) + spine_w)
            arr[:spine_x, y0:y1, :] = 1.0

            bar_w = max(1, int(round(nx * 0.03)))
            bar_x0 = max(0, spine_x - bar_w)
            bar_y0 = max(0, int(round(ny * 0.2)))
            bar_y1 = min(ny, int(round(ny * 0.8)))
            arr[bar_x0:spine_x, bar_y0:bar_y1, :] = 1.0
            add_feed_spine(arr, nx, ny, nz)
    elif mode == "loop":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        if nx < 5 or ny < 5:
            arr[:] = 1.0
        else:
            border = max(1, int(round(min(nx, ny) * 0.05)))
            x0 = max(0, int(round(nx * 0.1)))
            x1 = min(nx, int(round(nx * 0.9)))
            y0 = max(0, int(round(ny * 0.2)))
            y1 = min(ny, int(round(ny * 0.8)))
            if x1 - x0 <= 2 * border or y1 - y0 <= 2 * border:
                arr[:] = 1.0
            else:
                arr[x0:x1, y0:y0 + border, :] = 1.0
                arr[x0:x1, y1 - border:y1, :] = 1.0
                arr[x0:x0 + border, y0:y1, :] = 1.0
                arr[x1 - border:x1, y0:y1, :] = 1.0
                add_feed_spine(arr, nx, ny, nz)
    elif mode == "folded_dipole":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        if nx < 6 or ny < 6:
            arr[:] = 1.0
        else:
            bar_w = max(1, int(round(ny * 0.06)))
            gap = max(1, int(round(ny * 0.1)))
            y_mid = ny // 2
            y0 = max(0, y_mid - gap - bar_w)
            y1 = min(ny, y_mid + gap + bar_w)
            x0 = max(0, int(round(nx * 0.1)))
            x1 = min(nx, int(round(nx * 0.85)))
            if x1 <= x0 + 1 or y1 <= y0 + 2 * bar_w:
                arr[:] = 1.0
            else:
                arr[x0:x1, y0:y0 + bar_w, :] = 1.0
                arr[x0:x1, y1 - bar_w:y1, :] = 1.0
                conn_w = max(1, int(round(nx * 0.03)))
                arr[x1 - conn_w:x1, y0:y1, :] = 1.0
                add_feed_spine(arr, nx, ny, nz)
    elif mode == "meander":
        arr = np.zeros(shape, dtype=np.float32)
        nx, ny, nz = shape
        if nx < 6 or ny < 6:
            arr[:] = 1.0
        else:
            band_h = max(1, int(round(ny * 0.07)))
            gap = max(1, int(round(ny * 0.12)))
            y_mid = ny // 2
            y_positions = [
                max(0, y_mid - gap - band_h),
                max(0, y_mid - (band_h // 2)),
                min(ny - band_h, y_mid + gap),
            ]
            x1 = max(2, int(round(nx * 0.8)))
            for y0 in y_positions:
                y1 = min(ny, y0 + band_h)
                arr[:x1, y0:y1, :] = 1.0
            conn_w = max(1, int(round(nx * 0.05)))
            y0 = y_positions[0]
            y1 = min(ny, y_positions[1] + band_h)
            arr[:conn_w, y0:y1, :] = 1.0
            y2 = y_positions[1]
            y3 = min(ny, y_positions[2] + band_h)
            arr[x1 - conn_w:x1, y2:y3, :] = 1.0
            add_feed_spine(arr, nx, ny, nz)
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    return jnp.asarray(arr)


def _load_device_params(
    path: str,
    shape: tuple[int, int, int],
    allow_resample: bool = False,
    resample_method: str = "nearest",
) -> jax.Array:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "params" in data:
            arr = data["params"]
        elif "device_params" in data:
            arr = data["device_params"]
        else:
            raise ValueError(f"No params found in {path}")
    else:
        arr = data
    if arr.shape != shape:
        if not allow_resample:
            raise ValueError(f"Params shape {arr.shape} does not match device shape {shape}")
        arr = resample_params(arr, shape, method=resample_method)
    return jnp.asarray(arr.astype(np.float32))


def _file_sha1(path: str) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()[:12]


def _build_gap_port(
    model: ModelConfig,
    freqs: Iterable[float],
    resolution_m: float,
    switch: fdtdx.OnOffSwitch | None = None,
) -> Tuple[GapVoltageSource, fdtdx.PhasorDetector, fdtdx.PhasorDetector, float]:
    port_len_m = common.mm_to_m(model.source_gap_mm)
    port_width_m = common.mm_to_m(model.feed_width_mm)
    port_gap_m = common.mm_to_m(model.substrate_thickness_mm)

    sweep_center = 0.5 * (model.sweep_low_hz + model.sweep_high_hz)
    sweep_width = max(model.sweep_high_hz - model.sweep_low_hz, model.f0_hz * 0.2)

    source = GapVoltageSource(
        name="port_source",
        partial_real_shape=(port_len_m, port_width_m, port_gap_m),
        wave_character=fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / sweep_center),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_frequency=sweep_center,
            spectral_width=sweep_width,
        ),
        polarization_axis=2,
        amplitude=1.0,
    )

    waves = tuple(fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f) for f in freqs)
    port_v = fdtdx.PhasorDetector(
        name="port_v",
        partial_real_shape=(port_len_m, port_width_m, port_gap_m),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ez", "Hy", "Hz"),
        switch=switch or fdtdx.OnOffSwitch(),
    )
    pad_m = 0.0
    port_i_height_m = max(port_gap_m, 4.0 * resolution_m)
    port_i_width_m = port_width_m + 2.0 * pad_m
    port_i_len_m = port_len_m + 2.0 * pad_m
    port_i = fdtdx.PhasorDetector(
        name="port_i",
        partial_real_shape=(port_i_len_m, port_i_width_m, port_i_height_m),
        wave_characters=waves,
        reduce_volume=False,
        components=("Hx", "Hy", "Hz"),
        switch=switch or fdtdx.OnOffSwitch(),
    )
    return source, port_v, port_i, port_gap_m


def _build_port_flux_detector(
    model: ModelConfig,
    resolution_m: float,
    freqs: Iterable[float],
    switch: fdtdx.OnOffSwitch | None = None,
) -> fdtdx.PhasorDetector:
    waves = tuple(fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f) for f in freqs)
    width_m = common.mm_to_m(model.port_width_mm)
    height_m = common.mm_to_m(
        model.substrate_thickness_mm + model.port_air_above_mm + model.copper_thickness_mm
    )
    return fdtdx.PhasorDetector(
        name="port_flux",
        partial_real_shape=(resolution_m, width_m, height_m),
        wave_characters=waves,
        reduce_volume=False,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        switch=switch or fdtdx.OnOffSwitch(),
    )


def _build_flux_detectors(f0_hz: float, switch: fdtdx.OnOffSwitch | None = None) -> Dict[str, fdtdx.Detector]:
    wave = (fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f0_hz),)
    flux_detectors = {
        "flux_x_pos": fdtdx.PhasorDetector(
            name="flux_x_pos",
            partial_grid_shape=(1, None, None),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
        "flux_x_neg": fdtdx.PhasorDetector(
            name="flux_x_neg",
            partial_grid_shape=(1, None, None),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
        "flux_y_pos": fdtdx.PhasorDetector(
            name="flux_y_pos",
            partial_grid_shape=(None, 1, None),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
        "flux_y_neg": fdtdx.PhasorDetector(
            name="flux_y_neg",
            partial_grid_shape=(None, 1, None),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
        "flux_z_pos": fdtdx.PhasorDetector(
            name="flux_z_pos",
            partial_grid_shape=(None, None, 1),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
        "flux_z_neg": fdtdx.PhasorDetector(
            name="flux_z_neg",
            partial_grid_shape=(None, None, 1),
            wave_characters=wave,
            reduce_volume=False,
            switch=switch or fdtdx.OnOffSwitch(),
        ),
    }
    return flux_detectors


def _build_feed_flux_detector(
    model: ModelConfig,
    resolution_m: float,
    freqs: Iterable[float],
    switch: fdtdx.OnOffSwitch | None = None,
) -> fdtdx.PhasorDetector:
    waves = tuple(fdtdx.WaveCharacter(wavelength=fdtdx.constants.c / f) for f in freqs)
    width_m = common.mm_to_m(model.port_width_mm)
    height_m = common.mm_to_m(model.substrate_thickness_mm + model.port_air_above_mm + model.copper_thickness_mm)
    return fdtdx.PhasorDetector(
        name="feed_flux",
        partial_real_shape=(resolution_m, width_m, height_m),
        wave_characters=waves,
        reduce_volume=False,
        switch=switch or fdtdx.OnOffSwitch(),
    )


def _run_fdtd_with_params(
    arrays: fdtdx.ArrayContainer,
    objects: fdtdx.ObjectContainer,
    config: fdtdx.SimulationConfig,
    params: fdtdx.ParameterContainer,
    key: jax.Array,
    **transform_kwargs,
) -> Tuple[fdtdx.ObjectContainer, fdtdx.ArrayContainer]:
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key, **transform_kwargs)
    key, subkey = jax.random.split(key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=subkey)
    return objects, arrays


def run_one(
    model_name: str,
    quality: str,
    backend: str,
    init_mode: str,
    seed: int,
    run_root: str,
    force: bool,
    use_loss: bool,
    params_override: str | None = None,
    beta_override: float | None = None,
    params_resample: bool = False,
    params_resample_method: str = "nearest",
    time_scale: float = 1.0,
    port_calibration: str | None = None,
    calibration_metric: str = "auto",
    port_reference: str | None = None,
    rl_mode: str = "vi",
    port_vi_mode: str = "loop",
    save_port_vi: bool = False,
) -> Dict:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_CONFIGS[model_name]
    qual = common.resolve_quality(quality, model.base_resolution_mm)

    resolution_m = common.mm_to_m(qual.resolution_mm)
    resolution_mm = common.m_to_mm(resolution_m)
    adjusted_model = adjust_model_for_resolution(model, resolution_mm)
    if adjusted_model is not model:
        print(
            f"[{model.name}] adjusted source_gap_mm from {model.source_gap_mm:.3f} "
            f"to {adjusted_model.source_gap_mm:.3f} for dx={resolution_mm:.3f}mm"
        )
        model = adjusted_model
    travel_m = common.mm_to_m(model.outer_radius_mm + model.air_margin_mm)
    time_s = max(qual.time_cycles / model.f_low_hz, 2.0 * travel_m / fdtdx.constants.c)
    if time_scale <= 0:
        raise ValueError("time_scale must be > 0")
    time_s *= float(time_scale)

    sweep_freqs = np.asarray(model.sweep_freqs())
    band_freqs = np.asarray(model.band_freqs(points=11))
    freq_arr = np.unique(np.concatenate([sweep_freqs, band_freqs]))
    freq_arr.sort()

    cal_data = None
    cal_metric = None
    if port_calibration:
        cal_data = calibration.load_calibration(port_calibration)
        cal_data = calibration.interp_calibration(cal_data, freq_arr)
        cal_metric = calibration_metric
        if cal_metric == "auto":
            cal_metric = cal_data.get("metric") or "vi"
        if cal_metric not in {"vi", "flux", "thevenin"}:
            raise ValueError(f"Unknown calibration metric: {cal_metric}")

    ref_data = None
    if port_reference:
        ref_data = calibration.load_port_reference(port_reference)
        ref_data = calibration.interp_port_reference(ref_data, freq_arr)

    if rl_mode not in {"vi", "flux", "thevenin"}:
        raise ValueError(f"Unknown rl_mode: {rl_mode}")
    if port_vi_mode not in {"loop", "power"}:
        raise ValueError(f"Unknown port_vi_mode: {port_vi_mode}")
    if rl_mode == "thevenin" and ref_data is None:
        raise ValueError("rl_mode=thevenin requires --port-reference")
    if cal_metric == "thevenin" and ref_data is None:
        raise ValueError("calibration metric thevenin requires --port-reference")

    mesh_ok, mesh_msg = common.validate_mesh_constraints(
        resolution_mm=resolution_mm,
        min_trace_mm=model.min_trace_mm,
        min_gap_mm=model.min_gap_mm,
        feed_width_mm=model.feed_width_mm,
        gap_mm=model.source_gap_mm,
    )

    source, port_v, port_i, port_gap_len_m = _build_gap_port(model, freq_arr, resolution_m)
    flux_detectors = _build_flux_detectors(model.f0_hz)
    port_flux = _build_port_flux_detector(model, resolution_m, freq_arr)
    feed_flux = _build_feed_flux_detector(model, resolution_m, freq_arr)

    scene = geometry.build_scene(
        model,
        resolution_m=resolution_m,
        time_s=time_s,
        backend=backend,
        use_loss=use_loss,
        port_source=source,
        port_v_detector=port_v,
        port_i_detector=port_i,
        port_flux_detector=port_flux,
        flux_detectors=flux_detectors,
        feed_flux_detector=feed_flux,
    )

    cfg = fdtdx.SimulationConfig(
        time=time_s,
        resolution=resolution_m,
        backend=backend,
    )

    beta = float(beta_override) if beta_override is not None else float(model.projection_beta)

    run_payload = {
        "model": model.name,
        "model_cfg": asdict(model),
        "quality": quality,
        "backend": backend,
        "resolution_mm": resolution_mm,
        "time_s": time_s,
        "seed": seed,
        "init": init_mode,
        "beta": beta,
        "port_calibration": port_calibration,
        "port_calibration_metric": cal_metric,
        "port_reference": port_reference,
        "rl_mode": rl_mode,
    }
    if params_override:
        run_payload["params_hash"] = _file_sha1(params_override)
    run_id = common.hash_payload(run_payload)
    run_dir = os.path.join(run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    params_path = os.path.join(run_dir, "params.json")
    if not force and os.path.exists(os.path.join(run_dir, "metrics.json")):
        with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2, sort_keys=True)

    if not mesh_ok:
        metrics_out = {
            "model": model.name,
            "valid": False,
            "error": f"mesh_constraints_failed: {mesh_msg}",
        }
        post.save_metrics(os.path.join(run_dir, "metrics.json"), metrics_out)
        post.save_summary(os.path.join(run_dir, "summary.txt"), metrics_out)
        print(f"[{model.name}] mesh constraints failed: {mesh_msg}")
        print("Run output directory: ", run_id)
        return metrics_out

    key = jax.random.PRNGKey(seed)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=scene.object_list,
        config=cfg,
        constraints=scene.constraints,
        key=key,
    )

    port_v_obj = next(det for det in objects.detectors if det.name == "port_v")
    port_i_obj = next(det for det in objects.detectors if det.name == "port_i")
    port_v_slice = port_v_obj.grid_slice
    port_i_slice = port_i_obj.grid_slice

    device = next(o for o in objects.devices if o.name == scene.device.name)
    seed_mask = geometry.feed_seed_mask(model, device)
    if params_override:
        init_params = _load_device_params(
            params_override,
            device.matrix_voxel_grid_shape,
            allow_resample=params_resample,
            resample_method=params_resample_method,
        )
    else:
        init_params = _init_device_params(
            device.matrix_voxel_grid_shape,
            init_mode,
            rng=np.random.default_rng(seed),
        )
    init_params = jnp.maximum(init_params, seed_mask)
    params = dict(params)
    params[device.name] = init_params

    key, subkey = jax.random.split(key)
    objects_run, arrays_run = _run_fdtd_with_params(arrays, objects, config, params, subkey, beta=beta)

    design_indices = device(params[device.name], expand_to_sim_grid=True, beta=beta)
    design_indices_np = np.array(design_indices)

    invalid = False
    port_v_phasor = np.array(arrays_run.detector_states["port_v"]["phasor"][0])
    port_i_phasor = np.array(arrays_run.detector_states["port_i"]["phasor"][0])
    port_flux_phasor = np.array(arrays_run.detector_states["port_flux"]["phasor"][0])
    port_power = np.zeros(port_flux_phasor.shape[0], dtype=np.complex128)
    for idx in range(port_flux_phasor.shape[0]):
        ph = port_flux_phasor[idx]
        port_power[idx] = metrics.complex_power_through_plane(ph[0:3], ph[3:6], axis=0, resolution_m=resolution_m)
    port_v_mask = None
    port_i_mask = None
    if arrays_run.pec_mask is not None:
        port_v_mask = 1.0 - np.array(arrays_run.pec_mask[*port_v_slice])
        port_i_mask = 1.0 - np.array(arrays_run.pec_mask[*port_i_slice])
    V = metrics.compute_gap_voltage(
        port_v_phasor[:, 0],
        port_gap_len_m,
        mask=port_v_mask,
        method="mean",
    )
    I_loop = metrics.compute_loop_current(
        port_i_phasor[:, 1],
        port_i_phasor[:, 2],
        resolution_m,
        mask=port_i_mask,
        current_axis=0,
    )
    I_power = np.full_like(V, np.nan, dtype=np.complex128)
    valid_power = np.isfinite(port_power) & np.isfinite(V) & (np.abs(V) > 1e-18)
    if np.any(valid_power):
        I_power[valid_power] = 2.0 * np.conj(port_power[valid_power]) / np.conj(V[valid_power])
    if port_vi_mode == "power":
        I = I_power
    else:
        I = I_loop
    I_vert = metrics.compute_loop_current(
        port_i_phasor[:, 0],
        port_i_phasor[:, 1],
        resolution_m,
        mask=port_i_mask,
        current_axis=2,
    )
    p_vi = 0.5 * np.real(V * np.conj(I))
    flip = np.isfinite(p_vi) & (p_vi < 0)
    if np.any(flip) and port_vi_mode != "power":
        I = np.where(flip, -I, I)
        p_vi = 0.5 * np.real(V * np.conj(I))

    V_nomask = metrics.compute_gap_voltage(
        port_v_phasor[:, 0],
        port_gap_len_m,
        mask=None,
        method="mean",
    )
    I_nomask = metrics.compute_loop_current(
        port_i_phasor[:, 1],
        port_i_phasor[:, 2],
        resolution_m,
        mask=None,
        current_axis=0,
    )
    I_vert_nomask = metrics.compute_loop_current(
        port_i_phasor[:, 0],
        port_i_phasor[:, 1],
        resolution_m,
        mask=None,
        current_axis=2,
    )
    p_vi_nomask = 0.5 * np.real(V_nomask * np.conj(I_nomask))
    flip_nomask = np.isfinite(p_vi_nomask) & (p_vi_nomask < 0)
    if np.any(flip_nomask):
        I_nomask = np.where(flip_nomask, -I_nomask, I_nomask)
        p_vi_nomask = 0.5 * np.real(V_nomask * np.conj(I_nomask))

    V_raw = V.copy()
    I_raw = I.copy()
    zin_raw = np.full_like(V_raw, np.nan, dtype=np.complex128)
    vi_mask_raw = np.isfinite(V_raw) & np.isfinite(I_raw) & (np.abs(I_raw) > 1e-18)
    zin_raw[vi_mask_raw] = V_raw[vi_mask_raw] / I_raw[vi_mask_raw]
    s11_raw = metrics.s11_from_impedance(zin_raw)
    s11_db_raw = metrics.s11_db(s11_raw)
    rl_db_raw = -s11_db_raw

    zin_nomask = np.full_like(V_nomask, np.nan, dtype=np.complex128)
    vi_mask_nomask = np.isfinite(V_nomask) & np.isfinite(I_nomask) & (np.abs(I_nomask) > 1e-18)
    zin_nomask[vi_mask_nomask] = V_nomask[vi_mask_nomask] / I_nomask[vi_mask_nomask]
    s11_nomask = metrics.s11_from_impedance(zin_nomask)
    s11_db_nomask = metrics.s11_db(s11_nomask)
    rl_db_nomask = -s11_db_nomask

    zin_vert = np.full_like(V_raw, np.nan, dtype=np.complex128)
    vi_mask_vert = np.isfinite(V_raw) & np.isfinite(I_vert) & (np.abs(I_vert) > 1e-18)
    zin_vert[vi_mask_vert] = V_raw[vi_mask_vert] / I_vert[vi_mask_vert]
    s11_vert = metrics.s11_from_impedance(zin_vert)
    s11_db_vert = metrics.s11_db(s11_vert)
    rl_db_vert = -s11_db_vert

    zin_flux = np.full_like(V_raw, np.nan, dtype=np.complex128)
    vi_mask_flux = np.isfinite(V_raw) & np.isfinite(port_power) & (np.abs(port_power) > 1e-18)
    if np.any(vi_mask_flux):
        zin_flux[vi_mask_flux] = (V_raw[vi_mask_flux] * np.conj(V_raw[vi_mask_flux])) / (
            2.0 * np.conj(port_power[vi_mask_flux])
        )
    s11_flux = metrics.s11_from_impedance(zin_flux)
    s11_db_flux = metrics.s11_db(s11_flux)
    rl_db_flux = -s11_db_flux

    zin_thevenin = None
    s11_thevenin = None
    rl_db_thevenin = None
    if ref_data is not None:
        z_load, s11_thevenin = calibration.thevenin_s11(V, ref_data["v_open"], ref_data["i_short"])
        zin_thevenin = z_load
        rl_db_thevenin = -metrics.s11_db(s11_thevenin)

    s11_cal = None
    rl_db_cal = None
    zin_cal = None
    if cal_data is not None:
        if cal_metric == "vi":
            base_s11 = s11
        elif cal_metric == "flux":
            base_s11 = s11_flux
        else:
            if s11_thevenin is None:
                raise ValueError("thevenin calibration requires a thevenin S11")
            base_s11 = s11_thevenin
        s11_cal = calibration.apply_oneport_calibration(base_s11, cal_data["A"], cal_data["B"], cal_data["C"])
        s11_cal = np.where(np.isfinite(s11_cal), s11_cal, base_s11)
        rl_db_cal = -metrics.s11_db(s11_cal)
        denom = 1.0 - s11_cal
        zin_cal = np.full_like(s11_cal, np.nan + 1j * np.nan, dtype=np.complex128)
        valid_z = np.isfinite(denom) & (np.abs(denom) > 1e-18)
        zin_cal[valid_z] = 50.0 * (1.0 + s11_cal[valid_z]) / denom[valid_z]

    f0_idx = int(np.argmin(np.abs(freq_arr - model.f0_hz)))
    feed_phasor = np.array(arrays_run.detector_states["feed_flux"]["phasor"][0])
    feed_e_max = float(np.max(np.abs(feed_phasor[f0_idx, 0:3])))
    feed_h_max = float(np.max(np.abs(feed_phasor[f0_idx, 3:6])))
    pin_flux_arr = np.zeros(feed_phasor.shape[0], dtype=np.float64)
    pin_flux_raw_arr = np.zeros(feed_phasor.shape[0], dtype=np.float64)
    for idx in range(feed_phasor.shape[0]):
        ph = feed_phasor[idx]
        power = metrics.complex_power_through_plane(
            ph[0:3],
            ph[3:6],
            axis=0,
            resolution_m=resolution_m,
        )
        pin_flux_raw_arr[idx] = np.real(power)
        pin_flux_arr[idx] = abs(pin_flux_raw_arr[idx])
    pin_flux = float(pin_flux_arr[f0_idx])
    pin_flux_raw = float(pin_flux_raw_arr[f0_idx])

    p_vi_raw = p_vi.copy()
    scale_full = np.full_like(p_vi, np.nan, dtype=np.float64)
    pin_vi = float("nan")
    v0_abs = float("nan")
    i0_abs = float("nan")
    if np.isfinite(V).any() and np.isfinite(I).any():
        v0 = V[f0_idx]
        i0 = I[f0_idx]
        p0 = 0.5 * np.real(v0 * np.conj(i0))
        scale = np.ones_like(p_vi, dtype=np.float64)
        valid_scale = np.isfinite(p_vi) & (p_vi > 0) & np.isfinite(pin_flux_arr) & (pin_flux_arr > 0)
        if np.any(valid_scale) and port_vi_mode != "power":
            scale_full[valid_scale] = pin_flux_arr[valid_scale] / p_vi[valid_scale]
            scale[valid_scale] = np.clip(scale_full[valid_scale], 0.1, 10.0)
            I = I * scale
            p_vi = p_vi * scale
            i0 = I[f0_idx]
            p0 = 0.5 * np.real(v0 * np.conj(i0))
        if np.isfinite(v0):
            v0_abs = float(np.abs(v0))
        if np.isfinite(i0):
            i0_abs = float(np.abs(i0))
        if np.isfinite(p0):
            pin_vi = float(p0)

    I_full = np.full_like(I_raw, np.nan, dtype=np.complex128)
    valid_full = np.isfinite(scale_full) & np.isfinite(I_raw)
    if np.any(valid_full):
        I_full[valid_full] = I_raw[valid_full] * scale_full[valid_full]
    zin_full = np.full_like(V_raw, np.nan, dtype=np.complex128)
    vi_mask_full = np.isfinite(V_raw) & np.isfinite(I_full) & (np.abs(I_full) > 1e-18)
    zin_full[vi_mask_full] = V_raw[vi_mask_full] / I_full[vi_mask_full]
    s11_full = metrics.s11_from_impedance(zin_full)
    s11_db_full = metrics.s11_db(s11_full)
    rl_db_full = -s11_db_full

    zin = np.full_like(V, np.nan, dtype=np.complex128)
    vi_mask = np.isfinite(V) & np.isfinite(I) & (np.abs(I) > 1e-18)
    zin[vi_mask] = V[vi_mask] / I[vi_mask]
    s11 = metrics.s11_from_impedance(zin)
    s11_db = metrics.s11_db(s11)
    rl_db = -s11_db

    zin_f0 = zin[f0_idx] if f0_idx < zin.shape[0] else np.nan
    zin_f0_real = float(np.real(zin_f0)) if np.isfinite(zin_f0) else float("nan")
    zin_f0_imag = float(np.imag(zin_f0)) if np.isfinite(zin_f0) else float("nan")
    zin_cal_f0_real = float("nan")
    zin_cal_f0_imag = float("nan")
    if zin_cal is not None and f0_idx < zin_cal.shape[0]:
        zin_cal_f0 = zin_cal[f0_idx]
        if np.isfinite(zin_cal_f0):
            zin_cal_f0_real = float(np.real(zin_cal_f0))
            zin_cal_f0_imag = float(np.imag(zin_cal_f0))

    min_cycles = 4.0
    valid_mask = (time_s * freq_arr) >= min_cycles
    if not np.any(valid_mask):
        valid_mask = np.ones_like(freq_arr, dtype=bool)
    use_cal_for_rl = False
    if rl_db_cal is not None:
        if rl_mode == "vi" and cal_metric == "vi":
            use_cal_for_rl = True
        elif rl_mode == "flux" and cal_metric == "flux":
            use_cal_for_rl = True
        elif rl_mode == "thevenin" and cal_metric == "thevenin":
            use_cal_for_rl = True

    if rl_mode == "thevenin" and rl_db_thevenin is not None:
        rl_selected = rl_db_thevenin
    elif use_cal_for_rl:
        rl_selected = rl_db_cal
    elif rl_mode == "flux":
        rl_selected = rl_db_flux
    else:
        rl_selected = rl_db

    rl_db_metric = np.where(valid_mask, rl_selected, -1e6)
    rl_db_raw_metric = np.where(valid_mask, rl_db_raw, -1e6)
    rl_db_nomask_metric = np.where(valid_mask, rl_db_nomask, -1e6)
    rl_db_full_metric = np.where(valid_mask, rl_db_full, -1e6)
    rl_db_vert_metric = np.where(valid_mask, rl_db_vert, -1e6)
    rl_db_flux_metric = np.where(valid_mask, rl_db_flux, -1e6)
    rl_db_cal_metric = None
    if rl_db_cal is not None:
        rl_db_cal_metric = np.where(valid_mask, rl_db_cal, -1e6)

    band_summary = metrics.summarize_sweep(freq_arr, rl_db_metric, model.f_low_hz, model.f_high_hz)
    band_summary_raw = metrics.summarize_sweep(freq_arr, rl_db_raw_metric, model.f_low_hz, model.f_high_hz)
    band_summary_nomask = metrics.summarize_sweep(freq_arr, rl_db_nomask_metric, model.f_low_hz, model.f_high_hz)
    band_summary_full = metrics.summarize_sweep(freq_arr, rl_db_full_metric, model.f_low_hz, model.f_high_hz)
    band_summary_vert = metrics.summarize_sweep(freq_arr, rl_db_vert_metric, model.f_low_hz, model.f_high_hz)
    band_summary_flux = metrics.summarize_sweep(freq_arr, rl_db_flux_metric, model.f_low_hz, model.f_high_hz)
    band_summary_cal = None
    if rl_db_cal_metric is not None:
        band_summary_cal = metrics.summarize_sweep(freq_arr, rl_db_cal_metric, model.f_low_hz, model.f_high_hz)

    band_summary_thevenin = None
    if rl_db_thevenin is not None:
        band_summary_thevenin = metrics.summarize_sweep(
            freq_arr,
            np.where(valid_mask, rl_db_thevenin, -1e6),
            model.f_low_hz,
            model.f_high_hz,
        )

    band_mask = (freq_arr >= model.f_low_hz) & (freq_arr <= model.f_high_hz) & valid_mask
    if np.any(band_mask):
        rl_peak_in_band = float(np.max(rl_selected[band_mask]))
        f_peak_in_band = float(freq_arr[band_mask][int(np.argmax(rl_selected[band_mask]))])
        rl_peak_in_band_raw = float(np.max(rl_db_raw[band_mask]))
        f_peak_in_band_raw = float(freq_arr[band_mask][int(np.argmax(rl_db_raw[band_mask]))])
        rl_peak_in_band_nomask = float(np.max(rl_db_nomask[band_mask]))
        f_peak_in_band_nomask = float(freq_arr[band_mask][int(np.argmax(rl_db_nomask[band_mask]))])
        rl_peak_in_band_full = float(np.max(rl_db_full[band_mask]))
        f_peak_in_band_full = float(freq_arr[band_mask][int(np.argmax(rl_db_full[band_mask]))])
        rl_peak_in_band_vert = float(np.max(rl_db_vert[band_mask]))
        f_peak_in_band_vert = float(freq_arr[band_mask][int(np.argmax(rl_db_vert[band_mask]))])
        rl_peak_in_band_flux = float(np.max(rl_db_flux[band_mask]))
        f_peak_in_band_flux = float(freq_arr[band_mask][int(np.argmax(rl_db_flux[band_mask]))])
        rl_peak_in_band_cal = float(np.max(rl_db_cal[band_mask])) if rl_db_cal is not None else float("nan")
        f_peak_in_band_cal = (
            float(freq_arr[band_mask][int(np.argmax(rl_db_cal[band_mask]))]) if rl_db_cal is not None else float("nan")
        )
        rl_peak_in_band_thevenin = (
            float(np.max(rl_db_thevenin[band_mask])) if rl_db_thevenin is not None else float("nan")
        )
        f_peak_in_band_thevenin = (
            float(freq_arr[band_mask][int(np.argmax(rl_db_thevenin[band_mask]))])
            if rl_db_thevenin is not None
            else float("nan")
        )
    else:
        rl_peak_in_band = float("nan")
        f_peak_in_band = float("nan")
        rl_peak_in_band_raw = float("nan")
        f_peak_in_band_raw = float("nan")
        rl_peak_in_band_nomask = float("nan")
        f_peak_in_band_nomask = float("nan")
        rl_peak_in_band_full = float("nan")
        f_peak_in_band_full = float("nan")
        rl_peak_in_band_vert = float("nan")
        f_peak_in_band_vert = float("nan")
        rl_peak_in_band_flux = float("nan")
        f_peak_in_band_flux = float("nan")
        rl_peak_in_band_cal = float("nan")
        f_peak_in_band_cal = float("nan")
        rl_peak_in_band_thevenin = float("nan")
        f_peak_in_band_thevenin = float("nan")

    match_meta = None
    matched_metrics = {}
    match3_meta = None
    match3_metrics = {}
    try:
        match_meta = matching.calc_l_match(zin_f0, model.f0_hz)
    except Exception:
        match_meta = None

    if match_meta:
        zin_matched, s11_matched, s11_matched_db = matching.apply_l_match(
            zin, freq_arr, match_meta, z0=50.0
        )
        post.save_s11_csv(os.path.join(run_dir, "s11_matched.csv"), freq_arr, s11_matched_db, zin_matched)
        post.save_s11_plot(os.path.join(run_dir, "s11_matched.png"), freq_arr, s11_matched_db)
        post.save_smith_csv(os.path.join(run_dir, "s11_matched_smith.csv"), freq_arr, s11_matched)

        rl_matched_db = -s11_matched_db
        matched_summary = metrics.summarize_sweep(
            freq_arr,
            rl_matched_db,
            model.f_low_hz,
            model.f_high_hz,
        )
        bw_hz, bw_frac = matching.matched_bandwidth(
            freq_arr,
            rl_matched_db,
            model.f_low_hz,
            model.f_high_hz,
            rl_target=10.0,
        )
        series_val = float(match_meta["series_value"])
        shunt_val = float(match_meta["shunt_value"])
        matched_metrics = {
            "rl_min_in_band_matched_db": matched_summary["rl_min_in_band_db"],
            "rl_peak_in_band_matched_db": matched_summary["rl_peak_db"],
            "f_peak_in_band_matched_hz": matched_summary["f_peak_hz"],
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

    try:
        match3_meta = matching.calc_pi_match(
            zin,
            freq_arr,
            model.f0_hz,
            model.f_low_hz,
            model.f_high_hz,
            z0=50.0,
            l_range_nh=(0.5, 20.0),
            c_range_pf=(0.2, 10.0),
            samples=8,
        )
    except Exception:
        match3_meta = None

    if match3_meta:
        post.save_s11_csv(
            os.path.join(run_dir, "s11_matched3.csv"),
            freq_arr,
            match3_meta["s11_db"],
            match3_meta["zin"],
        )
        post.save_s11_plot(os.path.join(run_dir, "s11_matched3.png"), freq_arr, match3_meta["s11_db"])
        post.save_smith_csv(
            os.path.join(run_dir, "s11_matched3_smith.csv"),
            freq_arr,
            match3_meta["s11"],
        )
        rl_match3 = -match3_meta["s11_db"]
        band_mask = (freq_arr >= model.f_low_hz) & (freq_arr <= model.f_high_hz)
        if np.any(band_mask):
            rl_band = rl_match3[band_mask]
            freq_band = freq_arr[band_mask]
            finite = np.isfinite(rl_band)
            if np.any(finite):
                peak_idx = int(np.argmax(rl_band[finite]))
                match3_metrics.update(
                    {
                        "match3_rl_min_in_band_db": float(np.min(rl_band[finite])),
                        "match3_rl_peak_in_band_db": float(rl_band[finite][peak_idx]),
                        "match3_f_peak_in_band_hz": float(freq_band[finite][peak_idx]),
                    }
                )
            else:
                match3_metrics.update(
                    {
                        "match3_rl_min_in_band_db": float("nan"),
                        "match3_rl_peak_in_band_db": float("nan"),
                        "match3_f_peak_in_band_hz": float("nan"),
                    }
                )
        else:
            match3_metrics.update(
                {
                    "match3_rl_min_in_band_db": float("nan"),
                    "match3_rl_peak_in_band_db": float("nan"),
                    "match3_f_peak_in_band_hz": float("nan"),
                }
            )
        match3_metrics.update(
            {
                "match3_topology": match3_meta["topology"],
                "match3_shunt_in_type": match3_meta["shunt_in_type"],
                "match3_shunt_in_value": float(match3_meta["shunt_in_value"]),
                "match3_series_type": match3_meta["series_type"],
                "match3_series_value": float(match3_meta["series_value"]),
                "match3_shunt_out_type": match3_meta["shunt_out_type"],
                "match3_shunt_out_value": float(match3_meta["shunt_out_value"]),
                "match3_bandwidth_hz": float(match3_meta["match_bandwidth_hz"]),
                "match3_bandwidth_frac": float(match3_meta["match_bandwidth_frac"]),
            }
        )

    if rl_mode == "thevenin" and s11_thevenin is not None:
        s11_selected = s11_thevenin
    elif use_cal_for_rl and s11_cal is not None:
        s11_selected = s11_cal
    else:
        s11_selected = s11
    invalid_reasons: List[str] = []
    if not np.all(np.isfinite(s11_selected[band_mask])):
        invalid = True
        invalid_reasons.append("s11_nonfinite")
    s11_invalid_thresh = 1.1
    if np.any(s11_selected[band_mask] > s11_invalid_thresh):
        invalid = True
        invalid_reasons.append("s11_gt_1p1")
    if rl_mode == "flux":
        pin = pin_flux if np.isfinite(pin_flux) else float("nan")
    else:
        pin = abs(pin_vi) if np.isfinite(pin_vi) else float("nan")
    if not np.isfinite(pin) or pin <= 0:
        invalid = True
        invalid_reasons.append("pin_nonpositive")

    def _plane_power(name: str, axis: int, sign: float, positive: bool) -> float:
        phasor = np.array(arrays_run.detector_states[name]["phasor"][0, 0])
        if positive:
            return metrics.positive_real_power_through_plane(
                phasor[0:3],
                phasor[3:6],
                axis,
                resolution_m,
                sign=sign,
            )
        return metrics.real_power_through_plane(
            phasor[0:3],
            phasor[3:6],
            axis,
            resolution_m,
            sign=sign,
        )

    p_x_pos_signed = _plane_power("flux_x_pos", axis=0, sign=1.0, positive=False)
    p_x_neg_signed = _plane_power("flux_x_neg", axis=0, sign=-1.0, positive=False)
    p_y_pos_signed = _plane_power("flux_y_pos", axis=1, sign=1.0, positive=False)
    p_y_neg_signed = _plane_power("flux_y_neg", axis=1, sign=-1.0, positive=False)
    p_z_pos_signed = _plane_power("flux_z_pos", axis=2, sign=1.0, positive=False)
    p_z_neg_signed = _plane_power("flux_z_neg", axis=2, sign=-1.0, positive=False)

    p_x_pos = _plane_power("flux_x_pos", axis=0, sign=1.0, positive=True)
    p_x_neg = _plane_power("flux_x_neg", axis=0, sign=-1.0, positive=True)
    p_y_pos = _plane_power("flux_y_pos", axis=1, sign=1.0, positive=True)
    p_y_neg = _plane_power("flux_y_neg", axis=1, sign=-1.0, positive=True)
    p_z_pos = _plane_power("flux_z_pos", axis=2, sign=1.0, positive=True)
    p_z_neg = _plane_power("flux_z_neg", axis=2, sign=-1.0, positive=True)

    p_fwd = p_x_pos
    p_back = p_x_neg
    p_side = p_y_pos + p_y_neg + p_z_pos + p_z_neg
    p_rad_signed = p_x_pos_signed + p_x_neg_signed + p_y_pos_signed + p_y_neg_signed + p_z_pos_signed + p_z_neg_signed
    p_rad = p_fwd + p_back + p_side
    if not np.isfinite(p_rad) or p_rad < 0.0:
        p_rad = 0.0
    p_rad_eff = p_rad_signed
    if not np.isfinite(p_rad_eff) or p_rad_eff < 0.0:
        p_rad_eff = 0.0
    eff_tol = 1.05
    eff_ok = (
        np.isfinite(pin)
        and abs(pin) > 1e-12
        and np.isfinite(p_rad_eff)
        and p_rad_eff <= abs(pin) * eff_tol
    )
    eta_rad = p_rad_eff / abs(pin) if eff_ok else float("nan")
    fwd_frac = p_fwd / (p_rad + 1e-12)
    fwd_frac_db = metrics.gain_db(fwd_frac)
    if not eff_ok or not np.isfinite(eta_rad) or eta_rad < model.eta_min:
        invalid = True
        invalid_reasons.append("eta_rad")

    fb_db = metrics.fb_ratio_db(p_fwd, p_back)

    flux_probe = np.array(arrays_run.detector_states["flux_x_pos"]["phasor"][0, 0])
    flux_e_max = float(np.max(np.abs(flux_probe[0:3])))
    flux_h_max = float(np.max(np.abs(flux_probe[3:6])))

    valid_freq_min = float(np.min(freq_arr[valid_mask])) if np.any(valid_mask) else float("nan")

    metrics_out = {
        "model": model.name,
        "valid": not invalid,
        "invalid_reasons": invalid_reasons,
        "run_id": run_id,
        "run_dir": run_dir,
        "port_calibration_path": port_calibration or None,
        "port_calibration_metric": cal_metric,
        "port_reference_path": port_reference or None,
        "rl_mode": rl_mode,
        "f_peak_hz": band_summary["f_peak_hz"],
        "rl_peak_db": band_summary["rl_peak_db"],
        "rl_min_in_band_db": band_summary["rl_min_in_band_db"],
        "rl_peak_thevenin_db": band_summary_thevenin["rl_peak_db"] if band_summary_thevenin else float("nan"),
        "rl_min_in_band_thevenin_db": band_summary_thevenin["rl_min_in_band_db"]
        if band_summary_thevenin
        else float("nan"),
        "f_peak_raw_hz": band_summary_raw["f_peak_hz"],
        "rl_peak_raw_db": band_summary_raw["rl_peak_db"],
        "rl_min_in_band_raw_db": band_summary_raw["rl_min_in_band_db"],
        "f_peak_nomask_hz": band_summary_nomask["f_peak_hz"],
        "rl_peak_nomask_db": band_summary_nomask["rl_peak_db"],
        "rl_min_in_band_nomask_db": band_summary_nomask["rl_min_in_band_db"],
        "f_peak_full_hz": band_summary_full["f_peak_hz"],
        "rl_peak_full_db": band_summary_full["rl_peak_db"],
        "rl_min_in_band_full_db": band_summary_full["rl_min_in_band_db"],
        "f_peak_vert_hz": band_summary_vert["f_peak_hz"],
        "rl_peak_vert_db": band_summary_vert["rl_peak_db"],
        "rl_min_in_band_vert_db": band_summary_vert["rl_min_in_band_db"],
        "f_peak_flux_hz": band_summary_flux["f_peak_hz"],
        "rl_peak_flux_db": band_summary_flux["rl_peak_db"],
        "rl_min_in_band_flux_db": band_summary_flux["rl_min_in_band_db"],
        "f_peak_cal_hz": band_summary_cal["f_peak_hz"] if band_summary_cal else float("nan"),
        "rl_peak_cal_db": band_summary_cal["rl_peak_db"] if band_summary_cal else float("nan"),
        "rl_min_in_band_cal_db": band_summary_cal["rl_min_in_band_db"] if band_summary_cal else float("nan"),
        "rl_peak_in_band_db": rl_peak_in_band,
        "f_peak_in_band_hz": f_peak_in_band,
        "rl_peak_in_band_raw_db": rl_peak_in_band_raw,
        "f_peak_in_band_raw_hz": f_peak_in_band_raw,
        "rl_peak_in_band_nomask_db": rl_peak_in_band_nomask,
        "f_peak_in_band_nomask_hz": f_peak_in_band_nomask,
        "rl_peak_in_band_full_db": rl_peak_in_band_full,
        "f_peak_in_band_full_hz": f_peak_in_band_full,
        "rl_peak_in_band_vert_db": rl_peak_in_band_vert,
        "f_peak_in_band_vert_hz": f_peak_in_band_vert,
        "rl_peak_in_band_flux_db": rl_peak_in_band_flux,
        "f_peak_in_band_flux_hz": f_peak_in_band_flux,
        "rl_peak_in_band_cal_db": rl_peak_in_band_cal,
        "f_peak_in_band_cal_hz": f_peak_in_band_cal,
        "rl_peak_in_band_thevenin_db": rl_peak_in_band_thevenin,
        "f_peak_in_band_thevenin_hz": f_peak_in_band_thevenin,
        "valid_freq_min_hz": valid_freq_min,
        "valid_freq_min_cycles": min_cycles,
        "pin_w": pin,
        "pin_flux_w": pin_flux,
        "pin_flux_raw_w": pin_flux_raw if np.isfinite(pin_flux_raw) else float("nan"),
        "pin_flux_sign": float(np.sign(pin_flux_raw)) if np.isfinite(pin_flux_raw) else float("nan"),
        "pin_vi_w": pin_vi if np.isfinite(pin_vi) else float("nan"),
        "pin_vi_raw_w": float(p_vi_raw[f0_idx]) if np.isfinite(p_vi_raw[f0_idx]) else float("nan"),
        "pin_port_flux_w": float(np.real(port_power[f0_idx])) if np.isfinite(port_power[f0_idx]) else float("nan"),
        "scale_full_f0": float(scale_full[f0_idx]) if np.isfinite(scale_full[f0_idx]) else float("nan"),
        "zin_f0_real_ohm": zin_f0_real,
        "zin_f0_imag_ohm": zin_f0_imag,
        "zin_cal_f0_real_ohm": zin_cal_f0_real,
        "zin_cal_f0_imag_ohm": zin_cal_f0_imag,
        "zin_thevenin_f0_real_ohm": float(np.real(zin_thevenin[f0_idx]))
        if zin_thevenin is not None and f0_idx < zin_thevenin.shape[0] and np.isfinite(zin_thevenin[f0_idx])
        else float("nan"),
        "zin_thevenin_f0_imag_ohm": float(np.imag(zin_thevenin[f0_idx]))
        if zin_thevenin is not None and f0_idx < zin_thevenin.shape[0] and np.isfinite(zin_thevenin[f0_idx])
        else float("nan"),
        "scale_full_min": float(np.nanmin(scale_full)) if np.any(np.isfinite(scale_full)) else float("nan"),
        "scale_full_max": float(np.nanmax(scale_full)) if np.any(np.isfinite(scale_full)) else float("nan"),
        "v0_abs_v": v0_abs,
        "i0_abs_a": i0_abs,
        "feed_e_max": feed_e_max,
        "feed_h_max": feed_h_max,
        "flux_x_e_max": flux_e_max,
        "flux_x_h_max": flux_h_max,
        "prad_w": p_rad,
        "prad_signed_w": p_rad_signed,
        "prad_eff_w": p_rad_eff,
        "eta_rad": round(float(eta_rad), 6) if np.isfinite(eta_rad) else float("nan"),
        "p_fwd_w": p_fwd,
        "p_back_w": p_back,
        "p_side_w": p_side,
        "fwd_frac_db": round(float(fwd_frac_db), 3),
        "fb_db": round(fb_db, 3),
    }
    if matched_metrics:
        metrics_out.update(matched_metrics)
    if match3_metrics:
        metrics_out.update(match3_metrics)
    if save_port_vi:
        port_vi_path = os.path.join(run_dir, "port_vi.csv")
        with open(port_vi_path, "w", encoding="utf-8") as handle:
            handle.write("freq_hz,v_real,v_imag,i_real,i_imag\n")
            for f, v, i in zip(freq_arr, V, I):
                handle.write(f"{f},{np.real(v)},{np.imag(v)},{np.real(i)},{np.imag(i)}\n")
    post.save_s11_csv(os.path.join(run_dir, "s11.csv"), freq_arr, s11_db, zin)
    post.save_s11_plot(os.path.join(run_dir, "s11.png"), freq_arr, s11_db)
    post.save_s11_csv(os.path.join(run_dir, "s11_raw.csv"), freq_arr, s11_db_raw, zin_raw)
    post.save_s11_plot(os.path.join(run_dir, "s11_raw.png"), freq_arr, s11_db_raw)
    post.save_s11_csv(os.path.join(run_dir, "s11_nomask.csv"), freq_arr, s11_db_nomask, zin_nomask)
    post.save_s11_plot(os.path.join(run_dir, "s11_nomask.png"), freq_arr, s11_db_nomask)
    post.save_s11_csv(os.path.join(run_dir, "s11_full.csv"), freq_arr, s11_db_full, zin_full)
    post.save_s11_plot(os.path.join(run_dir, "s11_full.png"), freq_arr, s11_db_full)
    post.save_s11_csv(os.path.join(run_dir, "s11_vert.csv"), freq_arr, s11_db_vert, zin_vert)
    post.save_s11_plot(os.path.join(run_dir, "s11_vert.png"), freq_arr, s11_db_vert)
    post.save_s11_csv(os.path.join(run_dir, "s11_flux.csv"), freq_arr, s11_db_flux, zin_flux)
    post.save_s11_plot(os.path.join(run_dir, "s11_flux.png"), freq_arr, s11_db_flux)
    if s11_thevenin is not None:
        post.save_s11_csv(os.path.join(run_dir, "s11_thevenin.csv"), freq_arr, metrics.s11_db(s11_thevenin), zin_thevenin)
        post.save_s11_plot(os.path.join(run_dir, "s11_thevenin.png"), freq_arr, metrics.s11_db(s11_thevenin))
    if s11_cal is not None:
        s11_db_cal = metrics.s11_db(s11_cal)
        if zin_cal is None:
            denom = 1.0 - s11_cal
            zin_cal = np.full_like(s11_cal, np.nan + 1j * np.nan, dtype=np.complex128)
            valid_z = np.isfinite(denom) & (np.abs(denom) > 1e-18)
            zin_cal[valid_z] = 50.0 * (1.0 + s11_cal[valid_z]) / denom[valid_z]
        post.save_s11_csv(os.path.join(run_dir, "s11_cal.csv"), freq_arr, s11_db_cal, zin_cal)
        post.save_s11_plot(os.path.join(run_dir, "s11_cal.png"), freq_arr, s11_db_cal)
    post.save_metrics(os.path.join(run_dir, "metrics.json"), metrics_out)
    post.save_summary(os.path.join(run_dir, "summary.txt"), metrics_out)

    np.savez(
        os.path.join(run_dir, "geometry.npz"),
        params=np.array(init_params),
        design_indices=design_indices_np,
        resolution_mm=common.m_to_mm(resolution_m),
    )

    print(
        f"[{model.name}] f_peak={band_summary['f_peak_hz'] / 1e9:.3f} GHz "
        f"RL_peak={band_summary['rl_peak_db']:.2f} dB "
        f"RL_min_in_band={band_summary['rl_min_in_band_db']:.2f} dB "
        f"eta={eta_rad:.2f} FB={fb_db:.2f} dB"
    )
    print("Run output directory: ", run_id)
    return metrics_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--quality", choices=["coarse", "mid", "fast", "medium", "fine", "high"], default="coarse")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--init",
        choices=[
            "patch",
            "solid",
            "empty",
            "uniform",
            "yagi",
            "inverted_f",
            "loop",
            "folded_dipole",
            "meander",
            "seed_blob",
            "random_walk",
            "stochastic_grid",
        ],
        default="patch",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lossy", action="store_true")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--params", default=None, help="Optional device params .npy/.npz to override init")
    parser.add_argument("--params-resample", action="store_true")
    parser.add_argument("--params-resample-method", choices=["nearest", "linear"], default="nearest")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--port-calibration", default=None, help="Calibration JSON for port S11")
    parser.add_argument("--port-calibration-metric", choices=["vi", "flux", "thevenin", "auto"], default="auto")
    parser.add_argument("--port-reference", default=None, help="Reference JSON for thevenin S11")
    parser.add_argument("--rl-mode", choices=["vi", "flux", "thevenin"], default="vi")
    parser.add_argument("--save-port-vi", action="store_true", help="Write port_vi.csv with V/I sweeps")
    parser.add_argument(
        "--port-vi-mode",
        choices=["loop", "power"],
        default="loop",
        help="Select how port current is derived for V/I metrics (loop or power-based).",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_root = args.run_root or os.path.join(root, "runs")

    run_one(
        model_name=args.model,
        quality=args.quality,
        backend=args.backend,
        init_mode=args.init,
        seed=args.seed,
        run_root=run_root,
        force=args.force,
        use_loss=args.lossy,
        params_override=args.params,
        beta_override=args.beta,
        params_resample=args.params_resample,
        params_resample_method=args.params_resample_method,
        time_scale=args.time_scale,
        port_calibration=args.port_calibration,
        calibration_metric=args.port_calibration_metric,
        port_reference=args.port_reference,
        rl_mode=args.rl_mode,
        port_vi_mode=args.port_vi_mode,
        save_port_vi=args.save_port_vi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

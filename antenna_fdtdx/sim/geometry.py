"""Build FDTDX geometry for antenna simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

import fdtdx

from . import common
from .models import ModelConfig


@dataclass
class SceneObjects:
    volume: fdtdx.SimulationVolume
    substrate: fdtdx.ExtrudedPolygon
    ground: fdtdx.ExtrudedPolygon
    feed_line: fdtdx.UniformMaterialObject
    feed_stub: fdtdx.UniformMaterialObject
    device: fdtdx.Device
    source: fdtdx.Source
    port_v_detector: fdtdx.Detector
    port_i_detector: fdtdx.Detector
    port_flux_detector: fdtdx.Detector
    detectors: Dict[str, fdtdx.Detector]
    object_list: List
    constraints: List
    meta: Dict


def _make_materials(f0_hz: float, use_loss: bool) -> Dict[str, fdtdx.Material]:
    eps_r = 4.4
    tan_delta = 0.02
    if use_loss:
        eps0 = fdtdx.constants.eps0
        sigma = 2 * np.pi * f0_hz * eps0 * eps_r * tan_delta
    else:
        sigma = 0.0
    materials = {
        "Air": fdtdx.Material(permittivity=1.0),
        "FR4": fdtdx.Material(permittivity=eps_r, electric_conductivity=float(sigma)),
        "Copper": fdtdx.Material(permittivity=1.0, is_pec=True),
    }
    return materials


def _wedge_vertices(config: ModelConfig) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float, float, float]]:
    points = common.wedge_polygon_points(
        config.inner_radius_mm,
        config.outer_radius_mm,
        config.slice_angle_deg,
        n=48,
    )
    shifted, offset = common.normalize_polygon(points)
    bounds = common.polygon_bounds(shifted)
    vertices = np.asarray(shifted, dtype=float) * 1e-3
    return vertices, offset, bounds


def _ground_vertices(config: ModelConfig, offset: Tuple[float, float]) -> np.ndarray:
    trim_mm = max(config.ground_trim_mm, 0.0)
    min_outer_mm = config.inner_radius_mm + max(config.min_gap_mm, 0.5)
    outer_mm = max(min_outer_mm, config.outer_radius_mm - trim_mm)
    points = common.wedge_polygon_points(
        config.inner_radius_mm,
        outer_mm,
        config.slice_angle_deg,
        n=48,
    )
    shifted = [(x - offset[0], y - offset[1]) for x, y in points]
    return np.asarray(shifted, dtype=float) * 1e-3


def feed_seed_mask(model: ModelConfig, device: fdtdx.Device) -> jnp.ndarray:
    nx, ny, nz = device.matrix_voxel_grid_shape
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Device voxel grid shape must be positive")

    cell_x_mm = model.design_length_mm / nx
    cell_y_mm = model.design_width_mm / ny
    seed_len_mm = max(model.feed_stub_length_mm, cell_x_mm)
    seed_half_w_mm = max(0.5 * model.feed_width_mm, cell_y_mm)

    seed_len_cells = max(1, int(round(seed_len_mm / cell_x_mm)))
    seed_half_w_cells = max(1, int(round(seed_half_w_mm / cell_y_mm)))

    y_center = ny // 2
    y0 = max(0, y_center - seed_half_w_cells)
    y1 = min(ny, y_center + seed_half_w_cells)

    mask = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    if y1 > y0:
        mask = mask.at[0:seed_len_cells, y0:y1, :].set(1.0)
    return mask


def build_scene(
    config: ModelConfig,
    resolution_m: float,
    time_s: float,
    backend: str,
    use_loss: bool,
    port_source: fdtdx.Source,
    port_v_detector: fdtdx.Detector,
    port_i_detector: fdtdx.Detector,
    port_flux_detector: fdtdx.Detector,
    flux_detectors: Dict[str, fdtdx.Detector],
    feed_flux_detector: fdtdx.Detector,
    device_material: fdtdx.Material | None = None,
) -> SceneObjects:
    materials = _make_materials(config.f0_hz, use_loss=use_loss)
    device_materials = {
        "Air": materials["Air"],
        "Copper": device_material or materials["Copper"],
    }
    vertices, offset, bounds = _wedge_vertices(config)
    ground_vertices = _ground_vertices(config, offset)
    min_x, max_x, min_y, max_y = bounds
    wedge_w_mm = max_x - min_x
    wedge_h_mm = max_y - min_y
    total_x_mm = config.feed_offset_mm + config.feed_length_mm + config.design_length_mm
    if total_x_mm > wedge_w_mm:
        raise ValueError(f"Design exceeds wedge width: {total_x_mm:.2f} > {wedge_w_mm:.2f} mm")
    if config.design_width_mm > wedge_h_mm:
        raise ValueError(f"Design exceeds wedge height: {config.design_width_mm:.2f} > {wedge_h_mm:.2f} mm")

    feed_in_len_mm = config.feed_length_mm - config.feed_stub_length_mm
    if feed_in_len_mm <= 0:
        raise ValueError("feed_length_mm must exceed feed_stub_length_mm")
    feed_in_len_m = common.mm_to_m(feed_in_len_mm)

    air_margin_m = common.mm_to_m(config.air_margin_mm)
    substrate_h_m = common.mm_to_m(config.substrate_thickness_mm)
    copper_h_m = common.mm_to_m(
        max(config.copper_thickness_mm, common.m_to_mm(resolution_m))
    )

    air_above_m = air_margin_m
    air_below_m = air_margin_m * 0.6

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(
            common.mm_to_m(wedge_w_mm) + 2 * air_margin_m,
            common.mm_to_m(wedge_h_mm) + 2 * air_margin_m,
            air_below_m + substrate_h_m + copper_h_m + air_above_m,
        ),
        material=materials["Air"],
    )

    substrate = fdtdx.ExtrudedPolygon(
        name="substrate",
        materials=materials,
        material_name="FR4",
        axis=2,
        vertices=vertices,
        partial_real_shape=(
            common.mm_to_m(wedge_w_mm),
            common.mm_to_m(wedge_h_mm),
            substrate_h_m,
        ),
    )

    ground = fdtdx.ExtrudedPolygon(
        name="ground",
        materials=materials,
        material_name="Copper",
        axis=2,
        vertices=ground_vertices,
        partial_real_shape=(
            common.mm_to_m(wedge_w_mm),
            common.mm_to_m(wedge_h_mm),
            copper_h_m,
        ),
    )

    feed_line = fdtdx.UniformMaterialObject(
        name="feed_line",
        partial_real_shape=(
            common.mm_to_m(feed_in_len_mm),
            common.mm_to_m(config.feed_width_mm),
            copper_h_m,
        ),
        material=materials["Copper"],
    )

    feed_stub = None
    if config.feed_stub_length_mm > 0.0:
        feed_stub = fdtdx.UniformMaterialObject(
            name="feed_stub",
            partial_real_shape=(
                common.mm_to_m(config.feed_stub_length_mm),
                common.mm_to_m(config.feed_width_mm),
                copper_h_m,
            ),
            material=materials["Copper"],
        )

    device = fdtdx.Device(
        name="design",
        partial_real_shape=(
            common.mm_to_m(config.design_length_mm),
            common.mm_to_m(config.design_width_mm),
            copper_h_m,
        ),
        materials=device_materials,
        param_transforms=[
            fdtdx.GaussianSmoothing2D(std_discrete=2),
            fdtdx.TanhProjection(),
        ],
        partial_voxel_real_shape=(
            resolution_m,
            resolution_m,
            copper_h_m,
        ),
    )

    constraints: List = []
    objects: List = [volume, substrate, ground, feed_line, device, port_source]
    if feed_stub is not None:
        objects.append(feed_stub)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml")
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(list(bound_dict.values()))
    constraints.extend(bound_constraints)

    constraints.append(substrate.place_at_center(volume, axes=(0, 1)))
    constraints.append(
        substrate.place_relative_to(
            volume,
            axes=(2,),
            own_positions=-1,
            other_positions=-1,
            margins=air_below_m,
        )
    )

    constraints.append(ground.place_at_center(substrate, axes=(0, 1)))
    constraints.append(ground.place_below(substrate))

    constraints.append(feed_line.place_at_center(substrate, axes=1))
    constraints.append(
        feed_line.place_relative_to(
            substrate,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=common.mm_to_m(config.feed_offset_mm),
        )
    )
    constraints.append(feed_line.place_above(substrate))

    if feed_stub is not None:
        constraints.append(feed_stub.place_at_center(substrate, axes=1))
        constraints.append(
            feed_stub.place_relative_to(
                device,
                axes=0,
                own_positions=1,
                other_positions=-1,
            )
        )
        constraints.append(feed_stub.place_above(substrate))

    constraints.append(device.place_at_center(substrate, axes=1))
    constraints.append(
        device.place_relative_to(
            substrate,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=common.mm_to_m(config.feed_offset_mm + config.feed_length_mm),
        )
    )
    constraints.append(device.place_above(substrate))

    port_source_len_m = float(port_source.partial_real_shape[0])
    port_det_len_m = max(
        float(port_v_detector.partial_real_shape[0]),
        float(port_i_detector.partial_real_shape[0]),
        float(port_flux_detector.partial_real_shape[0]),
    )
    max_source_offset_m = max(0.0, feed_in_len_m - (port_source_len_m + port_det_len_m))
    max_port_offset_m = max(0.0, feed_in_len_m - port_det_len_m)
    source_offset_m = common.mm_to_m(max(config.source_offset_mm, 0.0))
    port_offset_m = common.mm_to_m(max(config.port_offset_mm, 0.0))
    if source_offset_m > max_source_offset_m:
        source_offset_m = max_source_offset_m
    if port_offset_m > max_port_offset_m:
        port_offset_m = max_port_offset_m
    min_port_offset_m = source_offset_m + port_source_len_m
    if port_offset_m < min_port_offset_m:
        port_offset_m = min_port_offset_m
    if port_offset_m > max_port_offset_m:
        port_offset_m = max_port_offset_m

    constraints.append(port_source.place_at_center(feed_line, axes=1))
    constraints.append(port_source.place_at_center(substrate, axes=2))
    constraints.append(
        port_source.place_relative_to(
            feed_line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=source_offset_m,
        )
    )

    constraints.append(port_v_detector.place_at_center(feed_line, axes=1))
    constraints.append(port_v_detector.place_at_center(substrate, axes=2))
    constraints.append(
        port_v_detector.place_relative_to(
            feed_line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset_m,
        )
    )

    constraints.append(port_i_detector.place_at_center(feed_line, axes=1))
    constraints.append(
        port_i_detector.place_relative_to(
            feed_line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset_m,
        )
    )
    constraints.append(
        port_i_detector.place_relative_to(
            substrate,
            axes=2,
            own_positions=-1,
            other_positions=-1,
            margins=0.0,
        )
    )

    constraints.append(port_flux_detector.place_at_center(feed_line, axes=1))
    constraints.append(port_flux_detector.place_at_center(substrate, axes=2))
    constraints.append(
        port_flux_detector.place_relative_to(
            feed_line,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            margins=port_offset_m,
        )
    )

    detectors = {
        "port_v": port_v_detector,
        "port_i": port_i_detector,
        "port_flux": port_flux_detector,
        "feed_flux": feed_flux_detector,
    }
    detectors.update(flux_detectors)
    for det in detectors.values():
        objects.append(det)

    feed_flux_anchor = feed_stub if feed_stub is not None else feed_line
    constraints.append(feed_flux_detector.place_at_center(feed_flux_anchor, axes=(0, 1)))
    constraints.append(
        feed_flux_detector.place_relative_to(
            substrate,
            axes=2,
            own_positions=-1,
            other_positions=-1,
            margins=0.0,
        )
    )

    monitor_margin_m = common.mm_to_m(config.monitor_margin_mm)
    monitor_margin_m = max(monitor_margin_m, resolution_m * 2.0)

    if "flux_x_pos" in flux_detectors:
        det = flux_detectors["flux_x_pos"]
        constraints.append(det.place_at_center(volume, axes=(1, 2)))
        constraints.append(det.same_size(volume, axes=(1, 2)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=0,
                own_positions=1,
                other_positions=1,
                margins=-monitor_margin_m,
            )
        )
    if "flux_x_neg" in flux_detectors:
        det = flux_detectors["flux_x_neg"]
        constraints.append(det.place_at_center(volume, axes=(1, 2)))
        constraints.append(det.same_size(volume, axes=(1, 2)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=0,
                own_positions=-1,
                other_positions=-1,
                margins=monitor_margin_m,
            )
        )
    if "flux_y_pos" in flux_detectors:
        det = flux_detectors["flux_y_pos"]
        constraints.append(det.place_at_center(volume, axes=(0, 2)))
        constraints.append(det.same_size(volume, axes=(0, 2)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=1,
                own_positions=1,
                other_positions=1,
                margins=-monitor_margin_m,
            )
        )
    if "flux_y_neg" in flux_detectors:
        det = flux_detectors["flux_y_neg"]
        constraints.append(det.place_at_center(volume, axes=(0, 2)))
        constraints.append(det.same_size(volume, axes=(0, 2)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=1,
                own_positions=-1,
                other_positions=-1,
                margins=monitor_margin_m,
            )
        )
    if "flux_z_pos" in flux_detectors:
        det = flux_detectors["flux_z_pos"]
        constraints.append(det.place_at_center(volume, axes=(0, 1)))
        constraints.append(det.same_size(volume, axes=(0, 1)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=2,
                own_positions=1,
                other_positions=1,
                margins=-monitor_margin_m,
            )
        )
    if "flux_z_neg" in flux_detectors:
        det = flux_detectors["flux_z_neg"]
        constraints.append(det.place_at_center(volume, axes=(0, 1)))
        constraints.append(det.same_size(volume, axes=(0, 1)))
        constraints.append(
            det.place_relative_to(
                volume,
                axes=2,
                own_positions=-1,
                other_positions=-1,
                margins=monitor_margin_m,
            )
        )

    meta = {
        "resolution_m": resolution_m,
        "time_s": time_s,
        "port_gap_len_m": common.mm_to_m(config.substrate_thickness_mm),
        "monitor_margin_m": monitor_margin_m,
        "source_offset_mm": common.m_to_mm(source_offset_m),
        "port_offset_mm": common.m_to_mm(port_offset_m),
    }

    return SceneObjects(
        volume=volume,
        substrate=substrate,
        ground=ground,
        feed_line=feed_line,
        feed_stub=feed_stub,
        device=device,
        source=port_source,
        port_v_detector=port_v_detector,
        port_i_detector=port_i_detector,
        port_flux_detector=port_flux_detector,
        detectors=detectors,
        object_list=objects,
        constraints=constraints,
        meta=meta,
    )

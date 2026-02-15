"""Parameter loading/resampling helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _axis_indices(src_len: int, dst_len: int) -> np.ndarray:
    if src_len == dst_len:
        return np.arange(src_len, dtype=np.int64)
    coords = np.linspace(0.0, src_len - 1.0, dst_len)
    return np.clip(np.rint(coords), 0, src_len - 1).astype(np.int64)


def _resample_nearest(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    ix = _axis_indices(arr.shape[0], target_shape[0])
    iy = _axis_indices(arr.shape[1], target_shape[1])
    iz = _axis_indices(arr.shape[2], target_shape[2])
    return arr[np.ix_(ix, iy, iz)]


def _resample_linear(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    try:
        import jax
        import jax.numpy as jnp

        resized = jax.image.resize(jnp.asarray(arr), target_shape, method="linear")
        return np.asarray(resized)
    except Exception:
        return _resample_nearest(arr, target_shape)


def resample_params(
    arr: np.ndarray,
    target_shape: Tuple[int, int, int],
    method: str = "nearest",
) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if method == "linear":
        return _resample_linear(arr, target_shape)
    if method == "nearest":
        return _resample_nearest(arr, target_shape)
    raise ValueError(f"Unknown resample method: {method}")

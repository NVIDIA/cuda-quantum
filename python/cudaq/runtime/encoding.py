# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.kernel_types import _KERNEL_ONLY_ERROR_MESSAGE
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.util import trace
import cupy as cp
import numpy as np

State = cudaq_runtime.State


def _simulation_complex_dtype():
    target = cudaq_runtime.get_target()
    if target.get_precision() == cudaq_runtime.SimulationPrecision.fp64:
        return np.complex128
    return np.complex64


def _next_power_of_two(n):
    if n <= 0:
        raise ValueError("amplitude_encode: input must be non-empty.")
    if n & (n - 1) == 0:
        return n
    return 1 << (n - 1).bit_length()


def _as_amplitude_vector(data, dtype):
    if isinstance(data, State):
        return np, np.asarray(data, dtype=dtype).ravel()

    if isinstance(data, cp.ndarray):
        if data.ndim != 1:
            raise ValueError("amplitude_encode: expected a 1D vector.")
        return cp, cp.asarray(data, dtype=dtype).ravel()

    if isinstance(data, (list, tuple)):
        return np, np.asarray(data, dtype=dtype).ravel()

    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("amplitude_encode: expected a 1D vector.")
        return np, np.asarray(data, dtype=dtype).ravel()

    raise TypeError(
        "amplitude_encode: data must be a list, 1D array, or cudaq.State.")


@trace.traced
def amplitude_encode(
    data: list | np.ndarray | State,
    *,
    pad: complex | float = 0,
) -> State:
    """
    Map classical features to a normalized quantum state by amplitude encoding.

    Encodes a classical vector as state amplitudes, optionally padding to the
    next power of two, then L2-normalizing and returning a :class:`State`.

    Args:
      data: Classical features as a list, NumPy/CuPy array, or existing
          :class:`State`.
      pad: Value used to pad when ``len(data)`` is not a power of two (default
          ``0`` for zero-padding to the nearest ``2^n``).

    Returns:
      :class:`State`: Normalized state vector suitable for simulation and
          ``cudaq.State.from_data`` workflows.

    # Example:
    ``state = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)``
    """
    dtype = _simulation_complex_dtype()
    xp, vec = _as_amplitude_vector(data, dtype)

    n = vec.size
    target_len = _next_power_of_two(n)
    if target_len != n:
        pad_values = xp.full(target_len - n, pad, dtype=vec.dtype)
        vec = xp.concatenate([vec, pad_values])

    norm = xp.linalg.norm(vec)
    if norm == 0:
        raise ValueError("amplitude_encode: cannot normalize a zero vector.")
    vec = vec / norm

    if xp is not np:
        vec = vec.astype(dtype, copy=False)
    else:
        vec = np.asarray(vec, dtype=dtype)

    return State.from_data(vec)


def angular_encode(q, angles, *, rotation='Y'):
    """
    Encode classical features as single-qubit rotation gates inside a kernel.

    Applies ``rx``, ``ry``, or ``rz`` (per ``rotation``) on each qubit in ``q``
    with the corresponding angle from ``angles``. Must be called from within an
    ``@cudaq.kernel``; host-side calls raise ``RuntimeError``.

    Args:
      q: A ``cudaq.qvector`` (or ``qview``) register to encode into.
      angles: A ``list[float]`` of rotation angles, one per qubit.
      rotation: Rotation axis: ``'X'``, ``'Y'``, or ``'Z'`` (default ``'Y'``).

    # Example:
    ``@cudaq.kernel``
    ``def kernel(angles: list[float]):``
    ``    q = cudaq.qvector(3)``
    ``    cudaq.angular_encode(q, angles, rotation='Y')``
    """
    raise RuntimeError(
        _KERNEL_ONLY_ERROR_MESSAGE.format("cudaq.angular_encode"))

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
import numpy as np

State = cudaq_runtime.State


def _cupy():
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


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

    cp = _cupy()
    if cp is not None and isinstance(data, cp.ndarray):
        if data.ndim != 1:
            raise ValueError("amplitude_encode: expected a 1D vector.")
        return cp, cp.asarray(data, dtype=dtype).ravel()

    if type(data).__module__ == 'cupy':
        raise ImportError("amplitude_encode: CuPy is required for CuPy arrays. "
                          "Install cupy-cuda13x or pass a NumPy array/list.")

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
    r"""
    Map classical features to a normalized quantum state by amplitude encoding.

    Amplitude encoding represents a classical feature vector as the amplitudes
    of a pure state in the computational basis. Given a length-:math:`d` vector
    :math:`\mathbf{x} = (x_0, \ldots, x_{d-1})` (real or complex), the
    procedure is:

    1. **Pad** to length :math:`N = 2^n` for the smallest :math:`n` with
       :math:`N \ge d`.
       The padded vector :math:`\mathbf{x}'` satisfies :math:`x'_i = x_i` for
       :math:`i < d` and :math:`x'_i = \texttt{pad}` for :math:`d \le i < N`.

    2. **Normalize** with the Euclidean (L2) norm (must be non-zero).
       Coefficients are :math:`\alpha_i = x'_i / \|\mathbf{x}'\|_2`.

    3. **Form the state** in the :math:`n`-qubit computational basis:
       :math:`|\psi\rangle = \sum_{i=0}^{N-1} \alpha_i |i\rangle`, where
       :math:`|i\rangle` is the basis ``ket`` with index :math:`i` in binary.

    The returned :class:`State` stores :math:`\alpha_i` in little-endian index
    order (consistent with ``cudaq.State`` / ``qvector(state)``). Real inputs are
    promoted to complex amplitudes with zero imaginary part before padding.

    Args:
      data: Classical features as a list, NumPy/CuPy array, or existing
          :class:`State` (re-normalized after any padding).
      pad: Value used to pad when ``len(data)`` is not a power of two (default
          ``0`` for zero-padding to the nearest ``2^n``).

    Returns:
      :class:`State`: Normalized state vector suitable for simulation and
          ``cudaq.State.from_data`` workflows.

    Raises:
      ValueError: If ``data`` is empty, has zero norm after padding, or is not
          a 1D vector.
      TypeError: If ``data`` has an unsupported type.

    See ``cudaq.contrib.examples`` for complete examples.
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


def _rotation_gate_name(rotation):
    if not isinstance(rotation, str):
        raise ValueError(
            "cudaq.contrib.angular_encode: rotation must be a string")
    key = rotation.upper()
    gates = {'X': 'rx', 'Y': 'ry', 'Z': 'rz'}
    if key not in gates:
        raise ValueError(
            f"cudaq.contrib.angular_encode: unsupported rotation '{rotation}' "
            "(expected 'X', 'Y', or 'Z')")
    return gates[key]


def _static_veq_size(q):
    from cudaq.kernel.quake_value import QuakeValue
    from cudaq.mlir.dialects import quake

    if not isinstance(q, QuakeValue):
        raise TypeError(
            "cudaq.contrib.angular_encode: qubits must be a kernel qalloc "
            "register")
    ty = q.mlirValue.type
    if quake.VeqType.isinstance(ty) and quake.VeqType.hasSpecifiedSize(ty):
        return quake.VeqType.getSize(ty)
    return None


def _angular_encode_builder(kernel, q, angles, *, rotation='Y'):
    from cudaq.kernel.quake_value import QuakeValue

    gate_name = _rotation_gate_name(rotation)
    rotate = getattr(kernel, gate_name)

    if isinstance(angles, (list, tuple)):
        q_size = _static_veq_size(q)
        if q_size is not None and len(angles) != q_size:
            raise ValueError(
                "cudaq.contrib.angular_encode: number of angles must match "
                "the number of qubits")
        for i, theta in enumerate(angles):
            rotate(theta, q[i])
        return

    if isinstance(angles, QuakeValue):

        def body(i):
            rotate(angles[i], q[i])

        stop = q.size()
        kernel.for_loop(0, stop, body)
        return

    raise TypeError(
        "cudaq.contrib.angular_encode: angles must be a list[float] or a "
        "kernel list argument")


def angular_encode(kernel_or_q, q_or_angles, angles=None, *, rotation='Y'):
    r"""
    Encode classical features as single-qubit rotation gates inside a kernel.

    Angular (rotation) encoding maps a classical angle vector
    (:math:`\theta_0`, :math:`\ldots`, :math:`\theta_{n-1}`) to an
    :math:`n`-qubit product state by applying one parameterized rotation per
    qubit. Starting from the :math:`n`-fold product of :math:`|0\rangle`, the
    encoded state is

    .. math::

       |\psi\rangle
       = \prod_{i=0}^{n-1} R_{\mathrm{axis}}(\theta_i)\,|0\rangle_i,

    where the product applies :math:`R_{\mathrm{axis}}(\theta_i)` on qubit
    :math:`i` and leaves other qubits unchanged. CUDA-Q uses the standard Pauli
    rotation convention

    .. math::

       R_P(\theta) = e^{-i \theta P / 2}, \quad P \in \{X, Y, Z\},

    implemented as ``rx(θ)``, ``ry(θ)``, or ``rz(θ)`` when
    ``rotation`` is ``'X'``, ``'Y'``, or ``'Z'`` respectively (default ``'Y'``).

    For example, ``rotation='Y'`` on qubit :math:`i` gives
    :math:`R_Y(\theta_i)|0\rangle = \cos(\theta_i/2)|0\rangle +
    \sin(\theta_i/2)|1\rangle`.

    The number of angles must match the number of qubits in ``q`` when the
    register size is known at compile time.

    Two call patterns are supported:

    * **Kernel language** (inside ``@cudaq.kernel``): intercepted by the
      compiler; host calls with ``(q, angles)`` raise ``RuntimeError``.
    * **Builder** (``cudaq.make_kernel()``): call
      ``angular_encode(kernel, qubits, angles, rotation='Y')`` to append
      ``rx``/``ry``/``rz`` gates to the circuit under construction.

    Args:
      kernel_or_q: In builder mode, the :class:`Kernel` from
          ``cudaq.make_kernel()``. In kernel language, the ``q`` register
          (handled by the compiler, not this Python function).
      q_or_angles: In builder mode, the ``qalloc`` register. In kernel language,
          the ``angles`` list.
      angles: Builder mode only — ``list[float]`` or a kernel ``list[float]``
          argument (:class:`QuakeValue`).
      rotation: Rotation axis: ``'X'``, ``'Y'``, or ``'Z'`` (default ``'Y'``;
          case-insensitive in builder mode).

    Raises:
      RuntimeError: Kernel-language host misuse with ``(q, angles)``.
      ValueError: Invalid ``rotation`` or angle/qubit count mismatch (builder).
      TypeError: Invalid builder arguments.

    See ``cudaq.contrib.examples`` for complete examples.
    """
    from cudaq.kernel.kernel_builder import PyKernel

    if isinstance(kernel_or_q, PyKernel):
        if angles is None:
            raise TypeError(
                "cudaq.contrib.angular_encode: builder mode expects "
                "(kernel, qubits, angles)")
        return _angular_encode_builder(kernel_or_q,
                                       q_or_angles,
                                       angles,
                                       rotation=rotation)

    raise RuntimeError(
        _KERNEL_ONLY_ERROR_MESSAGE.format("cudaq.contrib.angular_encode"))

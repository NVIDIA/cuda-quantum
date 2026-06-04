# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy
import math


def amplitude_encode(data, pad=None):
    """
    Encode classical data into a quantum state via amplitude encoding.

    Maps a classical feature vector to a normalized quantum state by
    encoding values as amplitudes:

    .. math::

        \\mathbf{x} \\mapsto \\sum_i x_i |i\\rangle

    If the input vector length is not a power of 2, the ``pad`` argument
    specifies the value used to pad the vector to the nearest power of 2.

    The returned state is always L2-normalized.

    Args:
        data (list | numpy.ndarray): Classical feature vector to encode.
            Can be real or complex valued.
        pad (float | int | None): Value to use for padding when the input
            length is not a power of 2. If ``None`` and the length is not
            a power of 2, a ``ValueError`` is raised.

    Returns:
        cudaq.State: A normalized quantum state with amplitudes
        proportional to the input data.

    Raises:
        ValueError: If ``data`` is empty.
        ValueError: If the input length is not a power of 2 and
            ``pad`` is ``None``.
        ValueError: If all values (including any padding) are zero.

    Examples:

        .. code-block:: python

            import cudaq

            # Encode a 4-element vector (already a power of 2).
            state = cudaq.amplitude_encode([0.5, 0.5, 0.5, 0.5])

            # Encode a 3-element vector with zero-padding to length 4.
            state = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)
            # state amplitudes: [0.5774, 0.5774, 0.5774, 0.0]
    """
    # Lazy import to avoid circular dependency at module load time.
    from cudaq import State

    arr = numpy.asarray(data, dtype=numpy.complex128).ravel()

    if arr.size == 0:
        raise ValueError("Input data must not be empty.")

    n = arr.size
    # Check if n is a power of 2.
    if n & (n - 1) != 0:
        if pad is None:
            raise ValueError(
                f"Input length {n} is not a power of 2. "
                "Provide a `pad` value to pad to the nearest power of 2, "
                "e.g. pad=0 for zero-padding.")
        next_pow2 = 1 << math.ceil(math.log2(n))
        padding = numpy.full(next_pow2 - n,
                             fill_value=pad,
                             dtype=numpy.complex128)
        arr = numpy.concatenate([arr, padding])

    # Normalize to unit L2 norm.
    norm = numpy.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError(
            "Cannot normalize a zero vector. "
            "Input data (including any padding) must contain "
            "at least one non-zero value.")
    arr = arr / norm

    return State.from_data(arr)


def angular_encode(kernel, qubits, data, num_qubits, rotation='Y'):
    """
    Encode classical data as single-qubit rotation angles (angular embedding).

    Each feature value ``x_i`` is applied as a rotation gate on qubit ``i``:

    .. math::

        x_i \\mapsto R(x_i)|0\\rangle

    This function appends gates to an existing kernel using the builder
    API, following the same convention as :func:`cudaq.kernels.hwe`.

    Args:
        kernel (:class:`Kernel`): The existing ``cudaq.Kernel`` to
            append to.
        qubits (:class:`qview`): Pre-allocated qubits to apply
            rotations to.
        data (list[float] | :class:`QuakeValue`): Classical feature
            values used as rotation angles, or a ``QuakeValue`` list
            for parameterized kernels.
        num_qubits (int): Number of qubits to encode into. Must be
            ``>= len(data)`` when ``data`` is a plain list.
        rotation (str): Rotation axis — ``'X'``, ``'Y'``, or ``'Z'``
            (case-insensitive). Defaults to ``'Y'``.

    Raises:
        ValueError: If ``rotation`` is not one of ``'X'``, ``'Y'``,
            or ``'Z'``.
        ValueError: If ``num_qubits`` < ``len(data)`` (when ``data``
            is a plain list).

    Examples:

        .. code-block:: python

            import cudaq

            # --- With concrete data ---
            data = [0.1, 0.2, 0.3]
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(3)
            cudaq.kernels.angular_encode(
                kernel, qubits, data, 3, rotation='Y')
            print(cudaq.draw(kernel))
            #      ╭─────────╮
            # q0 : ┤ ry(0.1) ├
            #      ├─────────┤
            # q1 : ┤ ry(0.2) ├
            #      ├─────────┤
            # q2 : ┤ ry(0.3) ├
            #      ╰─────────╯

            # --- With parameterized kernel ---
            kernel, params = cudaq.make_kernel(list)
            qubits = kernel.qalloc(3)
            cudaq.kernels.angular_encode(
                kernel, qubits, params, 3, rotation='Y')
            print(cudaq.draw(kernel, [0.1, 0.2, 0.3]))
    """
    rotation = rotation.upper()
    if rotation not in ('X', 'Y', 'Z'):
        raise ValueError(
            f"Invalid rotation axis '{rotation}'. "
            "Must be 'X', 'Y', or 'Z'.")

    # Validate length when data is a plain Python list / array.
    if isinstance(data, (list, numpy.ndarray)):
        if num_qubits < len(data):
            raise ValueError(
                f"num_qubits ({num_qubits}) must be >= len(data) "
                f"({len(data)}).")

    # Select the rotation gate method on the kernel.
    gate_map = {
        'X': kernel.rx,
        'Y': kernel.ry,
        'Z': kernel.rz,
    }
    gate = gate_map[rotation]

    # Apply one rotation per qubit.  When data is a QuakeValue list
    # (parameterized kernel) we rely on num_qubits for the loop bound
    # because len() is not available on QuakeValue.
    for i in range(num_qubits):
        gate(data[i], qubits[i])
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq


def amplitude_encode(data, pad=None):
    """
    Encode classical data into a quantum state via amplitude encoding.

    Maps a classical feature vector to a normalized quantum state by encoding
    values as amplitudes:

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

    **Examples:**

    .. code-block:: python

        import cudaq

        # Encode a 3-element feature vector with zero-padding
        state = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)
        # state contains amplitudes [0.5773, 0.5773, 0.5773, 0.0]

        # Encode a 4-element vector (already a power of 2, no padding)
        state = cudaq.amplitude_encode([1.0, 0.0, 0.0, 0.0])
    """
    # Convert input to numpy array
    arr = np.asarray(data, dtype=float)

    # Validate non-empty
    if arr.size == 0:
        raise ValueError("Input data must be non-empty.")

    original_len = arr.size

    # Compute the required length (next power of 2)
    target_len = 1 << (original_len - 1).bit_length()

    # Pad if the length is not a power of 2
    if original_len != target_len:
        if pad is None:
            raise ValueError(
                f"Input data length ({original_len}) is not a power of 2, "
                "but no `pad` value was provided. "
                "Specify `pad=<value>` to zero-pad to the nearest power of 2.")
        arr = np.pad(arr, (0, target_len - original_len),
                     constant_values=pad)

    # Check that not all values are zero
    norm = np.linalg.norm(arr)
    if np.isclose(norm, 0.0):
        raise ValueError(
            "Data (including any padding) is all zeros; cannot normalize.")

    # L2-normalize
    arr = arr / norm

    # Convert to complex dtype matching the target simulator precision,
    # then construct a cudaq.State
    return cudaq.State.from_data(np.array(arr, dtype=cudaq.complex()))


def angular_encode(kernel, qubits, data, rotation='Y'):
    """
    Encode classical data as qubit rotation angles (angular encoding).

    Applies single-qubit rotation gates to encode each feature value:

    .. math::

        x_i \\mapsto R_\\text{axis}(x_i) |0\\rangle

    This follows the same builder-API pattern as :func:`~cudaq.kernels.hwe`.
    It works with both :func:`~cudaq.make_kernel` (builder API) and inside
    :func:`@cudaq.kernel <cudaq.kernel>` decorated functions.

    Args:
        kernel (:class:`Kernel`): The existing ``cudaq.Kernel`` or
            ``KernelBuilder`` to append gates to.
        qubits (:class:`qview` or :class:`qvector`): The qubits to apply
            the rotation gates to.
        data (list[float] | numpy.ndarray): Classical feature values to
            encode. Must have the same length as the number of qubits.
        rotation (str): Rotation axis: ``'X'``, ``'Y'``, or ``'Z'``
            (case-insensitive). Defaults to ``'Y'``.

    Raises:
        ValueError: If ``data`` is empty.
        ValueError: If the length of ``data`` does not match the number
            of qubits.
        ValueError: If ``rotation`` is not ``'X'``, ``'Y'``, or ``'Z'``.

    **Examples:**

    .. code-block:: python

        import cudaq

        # Builder API (make_kernel)
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(3)
        cudaq.kernels.angular_encode(kernel, q, [0.1, 0.2, 0.3], rotation='Y')
        print(cudaq.draw(kernel))
        #      ╭─────────╮
        # q0 : ┤ ry(0.1) ├
        #      ├─────────┤
        # q1 : ┤ ry(0.2) ├
        #      ├─────────┤
        # q2 : ┤ ry(0.3) ├
        #      ╰─────────╯

        # Inside @cudaq.kernel decorator
        @cudaq.kernel
        def kernel(angles: list[float]):
            q = cudaq.qvector(3)
            cudaq.kernels.angular_encode(q, angles, rotation='Y')
    """
    # Determine number of qubits
    n_qubits = len(qubits)

    # Convert data to a flat array
    angles = np.asarray(data, dtype=float).ravel()

    # Validate
    if angles.size == 0:
        raise ValueError("Input data must be non-empty.")

    if angles.size != n_qubits:
        raise ValueError(
            f"Data length ({angles.size}) must equal the number of "
            f"qubits ({n_qubits}).")

    rotation = rotation.upper()
    if rotation not in ('X', 'Y', 'Z'):
        raise ValueError(
            f"Unsupported rotation axis '{rotation}'. "
            "Use 'X', 'Y', or 'Z'.")

    # Apply the appropriate rotation to each qubit
    gate = getattr(kernel, f'r{rotation.lower()}')
    for i in range(n_qubits):
        gate(angles[i], qubits[i])

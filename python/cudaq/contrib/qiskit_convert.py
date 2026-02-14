# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Functions to convert Qiskit circuits and OpenQASM files to CUDA-Q kernels.

This module provides interoperability between Qiskit and CUDA-Q,
allowing users to convert Qiskit QuantumCircuit objects and OpenQASM
files into CUDA-Q kernels for simulation and execution.

Note:
    This module requires ``qiskit`` to be installed separately::

        pip install qiskit
"""

import numpy as np

from ..kernel.kernel_builder import make_kernel


def _try_import_qiskit():
    """Import qiskit QuantumCircuit.

    Returns:
        The QuantumCircuit class from qiskit.

    Raises:
        ImportError: If qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise ImportError("This feature requires qiskit. "
                          "Install it with: pip install qiskit") from e
    return QuantumCircuit


def from_qasm(qasm_file):
    """Create a CUDA-Q kernel from an OpenQASM file.

    This function reads an OpenQASM file and converts it to a CUDA-Q kernel
    by first parsing it with Qiskit and then converting the resulting circuit.

    Args:
        qasm_file: Path to the OpenQASM file as a string.

    Returns:
        A CUDA-Q kernel equivalent to the OpenQASM circuit.

    Raises:
        ImportError: If qiskit is not installed.
        FileNotFoundError: If the QASM file does not exist.
        RuntimeError: If the QASM file cannot be parsed.

    Example:
        .. code-block:: python

            kernel = cudaq.contrib.from_qasm('/path/to/circuit.qasm')
            result = cudaq.sample(kernel)
    """
    QuantumCircuit = _try_import_qiskit()
    try:
        qiskit_circuit = QuantumCircuit.from_qasm_file(qasm_file)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Could not parse QASM file '{qasm_file}': {e}") from e

    return from_qiskit(qiskit_circuit)


def from_qiskit(qiskit_circuit):
    """Create a CUDA-Q kernel from a Qiskit QuantumCircuit.

    This function converts a Qiskit QuantumCircuit to an equivalent CUDA-Q
    kernel by mapping Qiskit gates to their CUDA-Q counterparts.

    Args:
        qiskit_circuit: A qiskit.QuantumCircuit instance.

    Returns:
        A CUDA-Q kernel equivalent to the input Qiskit circuit.

    Raises:
        ImportError: If qiskit is not installed.
        ValueError: If the circuit contains unsupported gates.

    Supported gates:
        - Single qubit: h, x, y, z, s, t, sdg, tdg, id (identity)
        - Two qubit: cx, cy, cz, ch, swap, rxx, rzz
        - Three qubit: ccx (Toffoli)
        - Parametric: rx, ry, rz, r1, u3, u, p (phase)
        - Controlled parametric: crx, cry, crz
        - Special: sx, sxdg (sqrt-X and adjoint), barrier, measure

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)

            kernel = cudaq.contrib.from_qiskit(qc)
            result = cudaq.sample(kernel)
    """
    # Ensure qiskit is available (validates input type indirectly)
    _try_import_qiskit()

    kernel = make_kernel()
    num_qubits = qiskit_circuit.num_qubits
    qubits = kernel.qalloc(num_qubits)

    for instruction in qiskit_circuit.data:
        op_name = instruction.operation.name
        params = [float(p) for p in instruction.operation.params]
        q_indices = [
            qiskit_circuit.find_bit(q).index for q in instruction.qubits
        ]

        # Single qubit gates
        if op_name == 'h':
            kernel.h(qubits[q_indices[0]])
        elif op_name == 'x':
            kernel.x(qubits[q_indices[0]])
        elif op_name == 'y':
            kernel.y(qubits[q_indices[0]])
        elif op_name == 'z':
            kernel.z(qubits[q_indices[0]])
        elif op_name == 's':
            kernel.s(qubits[q_indices[0]])
        elif op_name == 't':
            kernel.t(qubits[q_indices[0]])
        elif op_name == 'sdg':
            kernel.sdg(qubits[q_indices[0]])
        elif op_name == 'tdg':
            kernel.tdg(qubits[q_indices[0]])
        elif op_name == 'id':
            pass  # Identity gate, no operation needed

        # Two qubit gates
        elif op_name == 'cx':
            kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'cy':
            kernel.cy(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'cz':
            kernel.cz(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'ch':
            kernel.ch(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'swap':
            kernel.swap(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'rxx':
            kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
            kernel.rx(params[0], qubits[q_indices[0]])
            kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'rzz':
            kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
            kernel.rz(params[0], qubits[q_indices[0]])
            kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])

        # Toffoli gate (3-qubit) - use cx with list of controls
        elif op_name == 'ccx':
            kernel.cx([qubits[q_indices[0]], qubits[q_indices[1]]],
                      qubits[q_indices[2]])

        # Parametric single qubit rotations
        elif op_name == 'rx':
            kernel.rx(params[0], qubits[q_indices[0]])
        elif op_name == 'ry':
            kernel.ry(params[0], qubits[q_indices[0]])
        elif op_name == 'rz':
            kernel.rz(params[0], qubits[q_indices[0]])
        elif op_name == 'r1' or op_name == 'p':
            # r1 and p (phase) gates are equivalent
            kernel.r1(params[0], qubits[q_indices[0]])

        # Controlled parametric rotations
        elif op_name == 'crx':
            kernel.crx(params[0], qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'cry':
            kernel.cry(params[0], qubits[q_indices[0]], qubits[q_indices[1]])
        elif op_name == 'crz':
            kernel.crz(params[0], qubits[q_indices[0]], qubits[q_indices[1]])

        # U3 gate: U3(theta, phi, lambda) = Rz(lambda) Ry(theta) Rz(phi)
        # Note: Qiskit U3 params are (theta, phi, lambda)
        elif op_name == 'u3' or op_name == 'u':
            kernel.u3(params[0], params[1], params[2], qubits[q_indices[0]])

        # sqrt-X gate decomposition
        elif op_name == 'sx':
            kernel.r1(np.pi / 4, qubits[q_indices[0]])
            kernel.x(qubits[q_indices[0]])
            kernel.r1(np.pi / 4, qubits[q_indices[0]])
            kernel.x(qubits[q_indices[0]])
            kernel.rx(np.pi / 2, qubits[q_indices[0]])
        elif op_name == 'sxdg':
            kernel.r1(-np.pi / 4, qubits[q_indices[0]])
            kernel.x(qubits[q_indices[0]])
            kernel.r1(-np.pi / 4, qubits[q_indices[0]])
            kernel.x(qubits[q_indices[0]])
            kernel.rx(-np.pi / 2, qubits[q_indices[0]])

        # Barrier - no operation in CUDA-Q
        elif op_name == 'barrier':
            pass

        # Measurements
        elif op_name == 'measure':
            kernel.mz(qubits[q_indices[0]])

        else:
            raise ValueError(f"Gate '{op_name}' is not supported. "
                             f"Cannot convert Qiskit circuit to CUDA-Q kernel.")

    return kernel

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Functions to convert OpenQASM files to CUDA-Q kernels.

This module reads OpenQASM source files and produces CUDA-Q kernels. The
current implementation routes through Qiskit's QASM 2.0 loader and then
reuses :func:`cudaq.contrib.from_qiskit` for gate-by-gate translation.

Note:
    This module currently requires ``qiskit`` to be installed.
"""

from .qiskit_convert import from_qiskit


def _try_import_qiskit():
    """Import Qiskit `QuantumCircuit`.

    Returns:
        The `QuantumCircuit` class from Qiskit.

    Raises:
        ImportError: If Qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise ImportError("This feature requires Qiskit. "
                          "Install it with: `pip install qiskit`") from e
    return QuantumCircuit


def from_qasm(qasm_file):
    """Create a CUDA-Q kernel from an OpenQASM file.

    This function reads an OpenQASM file and converts it to a CUDA-Q kernel
    by first parsing it with Qiskit and then converting the resulting circuit.

    Args:
        `qasm_file`: Path to the OpenQASM file as a string.

    Returns:
        A CUDA-Q kernel equivalent to the OpenQASM circuit.

    Raises:
        ImportError: If Qiskit is not installed.
        FileNotFoundError: If the QASM file does not exist.
        RuntimeError: If the QASM file cannot be parsed.
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

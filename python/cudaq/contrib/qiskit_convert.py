# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Functions to convert Qiskit circuits and OpenQASM files to CUDA-Q kernels.

This module provides interoperability between Qiskit and CUDA-Q,
allowing users to convert Qiskit `QuantumCircuit` objects and OpenQASM
files into CUDA-Q kernels for simulation and execution.

Note:
    This module requires ``qiskit`` to be installed.
"""

from dataclasses import dataclass

import numpy as np

from ..kernel.kernel_builder import make_kernel

_CUSTOM_DEFINITION_EXPANSION_LIMIT = 10
_CONTROL_FLOW_OPERATION_NAMES = {
    "if_else", "for_loop", "while_loop", "switch_case"
}


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
        raise ImportError(
            "This feature requires Qiskit. Install it with: `pip install qiskit`"
        ) from e
    return QuantumCircuit


def _operation_name(operation):
    return operation.name.lower()


def _is_control_flow_operation(op_name):
    return op_name in _CONTROL_FLOW_OPERATION_NAMES


def _is_standard_gate(operation):
    return getattr(operation, "_standard_gate", None) is not None


def _is_qiskit_standard_operation(operation):
    if _is_standard_gate(operation):
        return True
    base_class = getattr(operation, "base_class", operation.__class__)
    module = getattr(base_class, "__module__", "")
    return module.startswith("qiskit.circuit.library.standard_gates")


def _has_quantum_circuit_definition(operation):
    definition = getattr(operation, "definition", None)
    return definition is not None and hasattr(definition, "data")


def _has_open_controls(operation):
    num_ctrl_qubits = getattr(operation, "num_ctrl_qubits", 0)
    if num_ctrl_qubits == 0:
        return False
    ctrl_state = getattr(operation, "ctrl_state", None)
    if ctrl_state is None:
        return False
    return ctrl_state != (1 << num_ctrl_qubits) - 1


@dataclass(frozen=True)
class _GateSpec:
    num_qubits: int | None
    num_clbits: int
    num_params: int
    emitter: object
    min_qubits: int | None = None


def _unsupported_gate_error(op_name):
    return ValueError(f"Gate '{op_name}' is not supported. "
                      f"Cannot convert Qiskit circuit to CUDA-Q kernel.")


def _invalid_gate_error(op_name, reason):
    return ValueError(f"Gate '{op_name}' has invalid shape: {reason}. "
                      f"Cannot convert Qiskit circuit to CUDA-Q kernel.")


def _params(operation, op_name):
    try:
        return [float(p) for p in operation.params]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Gate '{op_name}' has unsupported non-numeric parameters. "
            f"Cannot convert Qiskit circuit to CUDA-Q kernel.") from e


def _single_qubit(method_name):

    def emit(kernel, qubits, q_indices, params):
        getattr(kernel, method_name)(qubits[q_indices[0]])

    return emit


def _two_qubit(method_name):

    def emit(kernel, qubits, q_indices, params):
        getattr(kernel, method_name)(qubits[q_indices[0]], qubits[q_indices[1]])

    return emit


def _no_op(kernel, qubits, q_indices, params):
    pass


def _emit_cs_adj(kernel, qubits, q_indices, params):
    kernel.cs(qubits[q_indices[0]], qubits[q_indices[1]], isAdj=True)


def _emit_ct_adj(kernel, qubits, q_indices, params):
    kernel.ct(qubits[q_indices[0]], qubits[q_indices[1]], isAdj=True)


def _emit_rxx(kernel, qubits, q_indices, params):
    kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
    kernel.rx(params[0], qubits[q_indices[0]])
    kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_rzz(kernel, qubits, q_indices, params):
    kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])
    kernel.rz(params[0], qubits[q_indices[1]])
    kernel.cx(qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_ccx(kernel, qubits, q_indices, params):
    kernel.cx([qubits[q_indices[0]], qubits[q_indices[1]]],
              qubits[q_indices[2]])


def _emit_mcx(kernel, qubits, q_indices, params):
    controls = [qubits[index] for index in q_indices[:-1]]
    kernel.cx(controls, qubits[q_indices[-1]])


def _emit_rx(kernel, qubits, q_indices, params):
    kernel.rx(params[0], qubits[q_indices[0]])


def _emit_ry(kernel, qubits, q_indices, params):
    kernel.ry(params[0], qubits[q_indices[0]])


def _emit_rz(kernel, qubits, q_indices, params):
    kernel.rz(params[0], qubits[q_indices[0]])


def _emit_r1(kernel, qubits, q_indices, params):
    kernel.r1(params[0], qubits[q_indices[0]])


def _emit_crx(kernel, qubits, q_indices, params):
    kernel.crx(params[0], qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_cry(kernel, qubits, q_indices, params):
    kernel.cry(params[0], qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_crz(kernel, qubits, q_indices, params):
    kernel.crz(params[0], qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_cr1(kernel, qubits, q_indices, params):
    kernel.cr1(params[0], qubits[q_indices[0]], qubits[q_indices[1]])


def _emit_cu3(kernel, qubits, q_indices, params):
    kernel.cu3(params[0], params[1], params[2], qubits[q_indices[0]],
               qubits[q_indices[1]])


def _emit_u3(kernel, qubits, q_indices, params):
    kernel.u3(params[0], params[1], params[2], qubits[q_indices[0]])


def _emit_u2(kernel, qubits, q_indices, params):
    kernel.u3(np.pi / 2, params[0], params[1], qubits[q_indices[0]])


def _emit_sx(kernel, qubits, q_indices, params):
    kernel.r1(np.pi / 4, qubits[q_indices[0]])
    kernel.x(qubits[q_indices[0]])
    kernel.r1(np.pi / 4, qubits[q_indices[0]])
    kernel.x(qubits[q_indices[0]])
    kernel.rx(np.pi / 2, qubits[q_indices[0]])


def _emit_sxdg(kernel, qubits, q_indices, params):
    kernel.r1(-np.pi / 4, qubits[q_indices[0]])
    kernel.x(qubits[q_indices[0]])
    kernel.r1(-np.pi / 4, qubits[q_indices[0]])
    kernel.x(qubits[q_indices[0]])
    kernel.rx(-np.pi / 2, qubits[q_indices[0]])


DIRECT_OPERATION_SPECS = {
    "h":
        _GateSpec(1, 0, 0, _single_qubit("h")),
    "x":
        _GateSpec(1, 0, 0, _single_qubit("x")),
    "y":
        _GateSpec(1, 0, 0, _single_qubit("y")),
    "z":
        _GateSpec(1, 0, 0, _single_qubit("z")),
    "s":
        _GateSpec(1, 0, 0, _single_qubit("s")),
    "t":
        _GateSpec(1, 0, 0, _single_qubit("t")),
    "sdg":
        _GateSpec(1, 0, 0, _single_qubit("sdg")),
    "tdg":
        _GateSpec(1, 0, 0, _single_qubit("tdg")),
    "id":
        _GateSpec(1, 0, 0, _no_op),
    "cx":
        _GateSpec(2, 0, 0, _two_qubit("cx")),
    "cy":
        _GateSpec(2, 0, 0, _two_qubit("cy")),
    "cz":
        _GateSpec(2, 0, 0, _two_qubit("cz")),
    "ch":
        _GateSpec(2, 0, 0, _two_qubit("ch")),
    "swap":
        _GateSpec(2, 0, 0, _two_qubit("swap")),
    "cswap":
        _GateSpec(3, 0, 0,
                  lambda k, q, i, p: k.cswap(q[i[0]], q[i[1]], q[i[2]])),
    "cs":
        _GateSpec(2, 0, 0, _two_qubit("cs")),
    "ct":
        _GateSpec(2, 0, 0, _two_qubit("ct")),
    "csdg":
        _GateSpec(2, 0, 0, _emit_cs_adj),
    "ctdg":
        _GateSpec(2, 0, 0, _emit_ct_adj),
    "rxx":
        _GateSpec(2, 0, 1, _emit_rxx),
    "rzz":
        _GateSpec(2, 0, 1, _emit_rzz),
    "ccx":
        _GateSpec(3, 0, 0, _emit_ccx),
    "mcx":
        _GateSpec(None, 0, 0, _emit_mcx, min_qubits=2),
    "rx":
        _GateSpec(1, 0, 1, _emit_rx),
    "ry":
        _GateSpec(1, 0, 1, _emit_ry),
    "rz":
        _GateSpec(1, 0, 1, _emit_rz),
    "r1":
        _GateSpec(1, 0, 1, _emit_r1),
    "p":
        _GateSpec(1, 0, 1, _emit_r1),
    "phase":
        _GateSpec(1, 0, 1, _emit_r1),
    "u1":
        _GateSpec(1, 0, 1, _emit_r1),
    "crx":
        _GateSpec(2, 0, 1, _emit_crx),
    "cry":
        _GateSpec(2, 0, 1, _emit_cry),
    "crz":
        _GateSpec(2, 0, 1, _emit_crz),
    "cp":
        _GateSpec(2, 0, 1, _emit_cr1),
    "cu1":
        _GateSpec(2, 0, 1, _emit_cr1),
    "cphase":
        _GateSpec(2, 0, 1, _emit_cr1),
    "cu3":
        _GateSpec(2, 0, 3, _emit_cu3),
    "u3":
        _GateSpec(1, 0, 3, _emit_u3),
    "u":
        _GateSpec(1, 0, 3, _emit_u3),
    "u2":
        _GateSpec(1, 0, 2, _emit_u2),
    "sx":
        _GateSpec(1, 0, 0, _emit_sx),
    "sxdg":
        _GateSpec(1, 0, 0, _emit_sxdg),
    "barrier":
        _GateSpec(None, 0, 0, _no_op, min_qubits=0),
    "measure":
        _GateSpec(1, 1, 0, lambda k, q, i, p: k.mz(q[i[0]])),
    "reset":
        _GateSpec(1, 0, 0, _single_qubit("reset")),
}


def _validate_direct_operation(operation, op_name, q_indices, c_indices, spec):
    if spec.num_qubits is not None and len(q_indices) != spec.num_qubits:
        raise _invalid_gate_error(
            op_name, f"expected {spec.num_qubits} qubits, got {len(q_indices)}")
    if spec.min_qubits is not None and len(q_indices) < spec.min_qubits:
        raise _invalid_gate_error(
            op_name,
            f"expected at least {spec.min_qubits} qubits, got {len(q_indices)}")
    if len(c_indices) != spec.num_clbits:
        raise _invalid_gate_error(
            op_name,
            f"expected {spec.num_clbits} classical bits, got {len(c_indices)}")
    if len(operation.params) != spec.num_params:
        raise _invalid_gate_error(
            op_name,
            f"expected {spec.num_params} parameters, got {len(operation.params)}",
        )


def _emit_instruction(kernel, qubits, instruction, circuit, q_mapping, depth):
    if depth > _CUSTOM_DEFINITION_EXPANSION_LIMIT:
        raise ValueError("Could not expand custom gate definitions within "
                         f"{_CUSTOM_DEFINITION_EXPANSION_LIMIT} iterations.")

    operation = instruction.operation
    op_name = _operation_name(operation)
    q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
    c_indices = [circuit.find_bit(c).index for c in instruction.clbits]
    if q_mapping is not None:
        q_indices = [q_mapping[index] for index in q_indices]

    if _is_control_flow_operation(op_name):
        raise ValueError("cudaq.contrib.from_qiskit() does not support "
                         "classical control flow in Qiskit/OpenQASM inputs.")

    if op_name == "u0":
        raise ValueError("Gate 'u0' is not supported. Cannot convert Qiskit "
                         "circuit to CUDA-Q kernel.")

    if _has_open_controls(operation) and _has_quantum_circuit_definition(
            operation):
        # Qiskit represents open controls by deriving a definition that wraps
        # the closed-control operation with X gates on each open control.
        definition = operation.definition
        for nested_instruction in definition.data:
            _emit_instruction(kernel, qubits, nested_instruction, definition,
                              q_indices, depth + 1)
        return

    if not _is_qiskit_standard_operation(
            operation) and _has_quantum_circuit_definition(operation):
        definition = operation.definition
        for nested_instruction in definition.data:
            _emit_instruction(kernel, qubits, nested_instruction, definition,
                              q_indices, depth + 1)
        return

    if _emit_direct_operation(kernel, qubits, operation, op_name, q_indices,
                              c_indices):
        return

    if _is_qiskit_standard_operation(
            operation) or not _has_quantum_circuit_definition(operation):
        raise _unsupported_gate_error(op_name)

    definition = operation.definition
    for nested_instruction in definition.data:
        _emit_instruction(kernel, qubits, nested_instruction, definition,
                          q_indices, depth + 1)


def _emit_direct_operation(kernel, qubits, operation, op_name, q_indices,
                           c_indices):
    spec = DIRECT_OPERATION_SPECS.get(op_name)
    if spec is None:
        return False

    _validate_direct_operation(operation, op_name, q_indices, c_indices, spec)
    params = _params(operation, op_name)
    spec.emitter(kernel, qubits, q_indices, params)
    return True


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


def from_qiskit(qiskit_circuit):
    """Create a CUDA-Q kernel from a Qiskit `QuantumCircuit`.

    This function converts a Qiskit `QuantumCircuit` to an equivalent CUDA-Q
    kernel by mapping Qiskit gates to their CUDA-Q counterparts.

    Args:
        `qiskit_circuit`: A `Qiskit.QuantumCircuit` instance.

    Returns:
        A CUDA-Q kernel equivalent to the input Qiskit circuit.

    Raises:
        ImportError: If Qiskit is not installed.
        ValueError: If the circuit contains unsupported gates.

    Supported gates:
        - Single qubit: `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`, `id` (identity)
        - Two qubit: `cx`, `cy`, `cz`, `ch`, `swap`, `cs`, `ct`, `csdg`,
          `ctdg`, `rxx`, `rzz`
        - Multi qubit: `ccx` (Toffoli), `cswap`, `mcx`
        - Parametric: `rx`, `ry`, `rz`, `r1`, `u1`, `u2`, `u3`, `u`,
          `p` (phase), `phase`
        - Controlled parametric: `crx`, `cry`, `crz`, `cp`, `cu1`,
          `cphase`, `cu3`
        - Special: `sx`, `sxdg`, barrier, measure, reset
    """
    # Ensure Qiskit is available (validates input type indirectly)
    _try_import_qiskit()

    kernel = make_kernel()
    num_qubits = qiskit_circuit.num_qubits
    qubits = kernel.qalloc(num_qubits)

    for instruction in qiskit_circuit.data:
        _emit_instruction(kernel, qubits, instruction, qiskit_circuit, None, 0)

    return kernel

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Functions to convert Qiskit circuits to CUDA-Q kernels.

This module provides interoperability between Qiskit and CUDA-Q,
allowing users to convert Qiskit `QuantumCircuit` objects into CUDA-Q
kernels for simulation and execution.

Note:
    This module requires ``qiskit`` to be installed.
"""

import numpy as np

from ..kernel.kernel_builder import make_kernel


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


# --------------------------------------------------------------------------- #
# Gate handlers
#
# Each handler has signature `handler(kernel, qs, params)` where:
#   - `kernel` is the `PyKernel` being built,
#   - `qs` is the list of `QuakeValue` qubits this instruction acts on, in the
#     order Qiskit provides them,
#   - `params` is the list of float parameters for this instruction.
# --------------------------------------------------------------------------- #


def _noop(kernel, qs, params):
    pass


def _sx(kernel, qs, params):
    # SX = e^(iπ/4) · RX(π/2). Global phase has no effect on sampling.
    kernel.rx(np.pi / 2, qs[0])


def _sxdg(kernel, qs, params):
    kernel.rx(-np.pi / 2, qs[0])


def _u1(kernel, qs, params):
    kernel.r1(params[0], qs[0])


def _u2(kernel, qs, params):
    # u2(φ, λ) = u3(π/2, φ, λ)
    kernel.u3(np.pi / 2, params[0], params[1], qs[0])


def _u3(kernel, qs, params):
    kernel.u3(params[0], params[1], params[2], qs[0])


def _r(kernel, qs, params):
    # RGate(θ, φ) = exp(-iθ/2 (cos(φ)X + sin(φ)Y)) ≡ u3(θ, φ - π/2, π/2 - φ)
    theta, phi = params
    kernel.u3(theta, phi - np.pi / 2, np.pi / 2 - phi, qs[0])


def _csdg(kernel, qs, params):
    # CS† is the diagonal matrix (1, 1, 1, -i) = CR1(-π/2)
    kernel.cr1(-np.pi / 2, qs[0], qs[1])


def _cphase(kernel, qs, params):
    kernel.cr1(params[0], qs[0], qs[1])


def _cu3(kernel, qs, params):
    kernel.cu3(params[0], params[1], params[2], [qs[0]], qs[1])


def _cu(kernel, qs, params):
    # CU(θ, φ, λ, γ) applies an extra global phase γ when control = |1⟩,
    # which is equivalent to R1(γ) on the control before the controlled-U3.
    theta, phi, lam, gamma = params
    kernel.r1(gamma, qs[0])
    kernel.cu3(theta, phi, lam, [qs[0]], qs[1])


def _csx(kernel, qs, params):
    # H · S · H = SX, therefore H(t) · CS(c,t) · H(t) = CSX(c,t).
    kernel.h(qs[1])
    kernel.cs(qs[0], qs[1])
    kernel.h(qs[1])


def _ccx(kernel, qs, params):
    kernel.cx([qs[0], qs[1]], qs[2])


def _ccz(kernel, qs, params):
    kernel.h(qs[2])
    kernel.cx([qs[0], qs[1]], qs[2])
    kernel.h(qs[2])


def _cswap(kernel, qs, params):
    kernel.cswap([qs[0]], qs[1], qs[2])


def _iswap(kernel, qs, params):
    kernel.s(qs[0])
    kernel.s(qs[1])
    kernel.h(qs[0])
    kernel.cx(qs[0], qs[1])
    kernel.cx(qs[1], qs[0])
    kernel.h(qs[1])


def _dcx(kernel, qs, params):
    kernel.cx(qs[0], qs[1])
    kernel.cx(qs[1], qs[0])


def _rxx(kernel, qs, params):
    # CX conjugates X on control into X⊗X, so RXX = CX · RX(θ)_c · CX.
    theta = params[0]
    kernel.cx(qs[0], qs[1])
    kernel.rx(theta, qs[0])
    kernel.cx(qs[0], qs[1])


def _rzz(kernel, qs, params):
    # CX conjugates Z on target into Z⊗Z, so RZZ = CX · RZ(θ)_t · CX.
    theta = params[0]
    kernel.cx(qs[0], qs[1])
    kernel.rz(theta, qs[1])
    kernel.cx(qs[0], qs[1])


def _ryy(kernel, qs, params):
    # RX(π/2) maps Y → Z; rotate to Z basis, apply RZZ, rotate back.
    theta = params[0]
    kernel.rx(np.pi / 2, qs[0])
    kernel.rx(np.pi / 2, qs[1])
    _rzz(kernel, qs, [theta])
    kernel.rx(-np.pi / 2, qs[0])
    kernel.rx(-np.pi / 2, qs[1])


def _rzx(kernel, qs, params):
    # H on target maps Z⊗X into Z⊗Z: `RZX` = H_t · RZZ · H_t.
    theta = params[0]
    kernel.h(qs[1])
    _rzz(kernel, qs, [theta])
    kernel.h(qs[1])


def _ecr(kernel, qs, params):
    # ECR = RZX(π/4) · (X ⊗ I) · RZX(-π/4)
    _rzx(kernel, qs, [np.pi / 4])
    kernel.x(qs[0])
    _rzx(kernel, qs, [-np.pi / 4])


def _xx_plus_yy(kernel, qs, params):
    # Reproduces the `XXPlusYYGate` decomposition as defined in Qiskit.
    theta, beta = params
    kernel.rz(beta, qs[0])
    kernel.rz(-np.pi / 2, qs[1])
    kernel.rx(np.pi / 2, qs[1])
    kernel.rz(np.pi / 2, qs[1])
    kernel.s(qs[0])
    kernel.cx(qs[1], qs[0])
    kernel.ry(theta / 2, qs[1])
    kernel.ry(theta / 2, qs[0])
    kernel.cx(qs[1], qs[0])
    kernel.sdg(qs[0])
    kernel.rz(-np.pi / 2, qs[1])
    kernel.rx(-np.pi / 2, qs[1])
    kernel.rz(np.pi / 2, qs[1])
    kernel.rz(-beta, qs[0])


def _xx_minus_yy(kernel, qs, params):
    # Reproduces the `XXMinusYYGate` decomposition as defined in Qiskit.
    theta, beta = params
    kernel.rz(-beta, qs[1])
    kernel.rz(-np.pi / 2, qs[0])
    kernel.rx(np.pi / 2, qs[0])
    kernel.rz(np.pi / 2, qs[0])
    kernel.s(qs[1])
    kernel.cx(qs[0], qs[1])
    kernel.ry(theta / 2, qs[0])
    kernel.ry(-theta / 2, qs[1])
    kernel.cx(qs[0], qs[1])
    kernel.sdg(qs[1])
    kernel.rz(-np.pi / 2, qs[0])
    kernel.rx(-np.pi / 2, qs[0])
    kernel.rz(np.pi / 2, qs[0])
    kernel.rz(beta, qs[1])


def _rccx(kernel, qs, params):
    # Relative-phase Toffoli (also called the `Margolus` gate, exposed as
    # `RCCXGate` in Qiskit).
    kernel.h(qs[2])
    kernel.t(qs[2])
    kernel.cx(qs[1], qs[2])
    kernel.tdg(qs[2])
    kernel.cx(qs[0], qs[2])
    kernel.t(qs[2])
    kernel.cx(qs[1], qs[2])
    kernel.tdg(qs[2])
    kernel.h(qs[2])


def _mcx(kernel, qs, params):
    # All qubits except the last are controls.
    kernel.cx(qs[:-1], qs[-1])


def _mcp(kernel, qs, params):
    kernel.cr1(params[0], qs[:-1], qs[-1])


_GATE_HANDLERS = {
    # Identity / timing / visual-only
    'id': _noop,
    'i': _noop,
    'barrier': _noop,
    'delay': _noop,
    'global_phase': _noop,

    # Paulis
    'x': lambda k, qs, p: k.x(qs[0]),
    'y': lambda k, qs, p: k.y(qs[0]),
    'z': lambda k, qs, p: k.z(qs[0]),

    # 1-qubit Clifford
    'h': lambda k, qs, p: k.h(qs[0]),
    's': lambda k, qs, p: k.s(qs[0]),
    'sdg': lambda k, qs, p: k.sdg(qs[0]),
    't': lambda k, qs, p: k.t(qs[0]),
    'tdg': lambda k, qs, p: k.tdg(qs[0]),
    'sx': _sx,
    'sxdg': _sxdg,

    # Phase / universal single-qubit
    'p': lambda k, qs, p: k.r1(p[0], qs[0]),
    'phase': lambda k, qs, p: k.r1(p[0], qs[0]),
    'u1': _u1,
    'u2': _u2,
    'u3': _u3,
    'u': _u3,
    'r': _r,

    # Rotations
    'rx': lambda k, qs, p: k.rx(p[0], qs[0]),
    'ry': lambda k, qs, p: k.ry(p[0], qs[0]),
    'rz': lambda k, qs, p: k.rz(p[0], qs[0]),

    # Two-qubit controlled
    'cx': lambda k, qs, p: k.cx(qs[0], qs[1]),
    'cnot': lambda k, qs, p: k.cx(qs[0], qs[1]),
    'cy': lambda k, qs, p: k.cy(qs[0], qs[1]),
    'cz': lambda k, qs, p: k.cz(qs[0], qs[1]),
    'ch': lambda k, qs, p: k.ch(qs[0], qs[1]),
    'cs': lambda k, qs, p: k.cs(qs[0], qs[1]),
    'csdg': _csdg,
    'cp': _cphase,
    'cphase': _cphase,
    'cu1': _cphase,
    'cu3': _cu3,
    'cu': _cu,
    'crx': lambda k, qs, p: k.crx(p[0], qs[0], qs[1]),
    'cry': lambda k, qs, p: k.cry(p[0], qs[0], qs[1]),
    'crz': lambda k, qs, p: k.crz(p[0], qs[0], qs[1]),
    'csx': _csx,

    # Swaps and exchange
    'swap': lambda k, qs, p: k.swap(qs[0], qs[1]),
    'iswap': _iswap,
    'dcx': _dcx,
    'ecr': _ecr,
    'cswap': _cswap,
    'fredkin': _cswap,

    # Two-qubit parametric
    'rxx': _rxx,
    'ryy': _ryy,
    'rzz': _rzz,
    'rzx': _rzx,
    'xx_plus_yy': _xx_plus_yy,
    'xx_minus_yy': _xx_minus_yy,

    # Three-qubit and multi-qubit
    'ccx': _ccx,
    'toffoli': _ccx,
    'rccx': _rccx,
    'ccz': _ccz,
    'c3x': _mcx,
    'c4x': _mcx,
    'mcx': _mcx,
    'mcx_gray': _mcx,
    'mcx_recursive': _mcx,
    'mcphase': _mcp,
    'mcp': _mcp,

    # Measurement / reset
    'measure': lambda k, qs, p: k.mz(qs[0]),
    'reset': lambda k, qs, p: k.reset(qs[0]),
}


def _apply_instruction(kernel, qs, operation):
    """Apply a Qiskit `Instruction` on the given CUDA-Q qubits.

    Dispatches by `operation.name` using `_GATE_HANDLERS`. For gates not in
    the table, recursively descends into `operation.definition` — this
    covers custom gates, controlled gates with non-default `ctrl_state`,
    `initialize`, and any gate that Qiskit expresses via its standard
    decomposition library.

    Returns True on success; False if the gate (and its decomposition) cannot
    be handled.
    """
    try:
        params = [float(p) for p in operation.params]
    except (TypeError, ValueError):
        # Unbound `ParameterExpression` or otherwise non-numeric parameters.
        return False

    handler = _GATE_HANDLERS.get(operation.name)
    if handler is not None:
        handler(kernel, qs, params)
        return True

    definition = getattr(operation, 'definition', None)
    if definition is None or len(definition.data) == 0:
        return False

    for sub_instruction in definition.data:
        sub_op = sub_instruction.operation
        sub_local = [
            definition.find_bit(q).index for q in sub_instruction.qubits
        ]
        sub_qs = [qs[i] for i in sub_local]
        if not _apply_instruction(kernel, sub_qs, sub_op):
            return False
    return True


def from_qiskit(qiskit_circuit):
    """Create a CUDA-Q kernel from a Qiskit `QuantumCircuit`.

    Each instruction is dispatched to a direct CUDA-Q equivalent when one
    exists, or to a decomposition built from supported CUDA-Q gates
    otherwise. Gates not listed below fall back to recursive expansion via
    the `Instruction.definition` attribute provided by Qiskit, so custom
    and composite gates are also supported as long as they bottom out in
    known primitives.

    Args:
        `qiskit_circuit`: A `Qiskit.QuantumCircuit` instance.

    Returns:
        A CUDA-Q kernel equivalent to the input Qiskit circuit.

    Raises:
        ImportError: If Qiskit is not installed.
        ValueError: If the circuit contains a gate with no direct mapping and
            no expandable `definition`.

    Directly supported gates:
        - Identity / no-op: ``id``, ``i``, ``barrier``, ``delay``, ``global_phase``
        - Pauli: ``x``, ``y``, ``z``
        - 1-qubit Clifford: ``h``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``, ``sxdg``
        - Phase / universal: ``p``/``phase``, ``u1``, ``u2``, ``u3``, ``u``, ``r``
        - Rotations: ``rx``, ``ry``, ``rz``
        - Controlled 1-qubit: ``cx``/``cnot``, ``cy``, ``cz``, ``ch``, ``cs``,
          ``csdg``, ``csx``, ``crx``, ``cry``, ``crz``, ``cp``/``cphase``,
          ``cu1``, ``cu3``, ``cu``
        - Swaps / exchange: ``swap``, ``iswap``, ``dcx``, ``ecr``,
          ``cswap``/``fredkin``
        - Two-qubit parametric: ``rxx``, ``ryy``, ``rzz``, ``rzx``,
          ``xx_plus_yy``, ``xx_minus_yy``
        - Multi-qubit: ``ccx``/``toffoli``, ``rccx``, ``ccz``, ``c3x``, ``c4x``,
          ``mcx`` (and variants), ``mcp``/``mcphase``
        - Measurement / reset: ``measure``, ``reset``
    """
    _try_import_qiskit()

    kernel = make_kernel()
    qubits = kernel.qalloc(qiskit_circuit.num_qubits)

    for instruction in qiskit_circuit.data:
        operation = instruction.operation
        qs = [
            qubits[qiskit_circuit.find_bit(q).index] for q in instruction.qubits
        ]

        if not _apply_instruction(kernel, qs, operation):
            raise ValueError(f"Gate '{operation.name}' is not supported. "
                             f"Cannot convert Qiskit circuit to CUDA-Q kernel.")

    return kernel

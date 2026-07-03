# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit cuDensityMat dialect MLIR text from an OperatorProgram.

**Reference implementation** -- the production path uses the C++ MLIR
lowering passes ``--pulse-to-qop`` then ``--qop-to-cudm``.  This Python
emitter is kept for testing and as documentation. It is used when
``CUDAQ_PULSE_LEGACY_PYTHON_PATH=1`` is set.

Generates the MLIR operations needed to create a quantum state,
build the system Hamiltonian and Lindblad collapse operators,
evolve the state, and extract the result.
"""

from __future__ import annotations

import textwrap
from typing import Any, List

from .pulse_to_operator import OperatorProgram, OperatorTerm


def emit_cudm_mlir(
    op_program: OperatorProgram,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
) -> str:
    """Convert an OperatorProgram to cuDensityMat dialect MLIR text.

    Parameters
    ----------
    op_program : OperatorProgram
        The operator program from ``pulse_to_operator``.
    t_start, t_end : float
        Time window in nanoseconds.
    num_steps : int
        Number of integration steps.
    integrator : str
        Integrator method (e.g. ``"magnus_cf4"``).

    Returns
    -------
    str
        MLIR module text in the cuDensityMat dialect.
    """
    if not isinstance(op_program, OperatorProgram):
        raise TypeError(
            f"Expected OperatorProgram, got {type(op_program).__name__}. "
            "Run pulse_to_operator first.")

    n_qubits = op_program.n_qubits
    hilbert_dim = 2**n_qubits

    lines: list[str] = []
    lines.append(f'module @{op_program.name}_cudm {{')
    lines.append(
        f'  func.func @evolve() -> memref<{hilbert_dim}xcomplex<f64>> {{')

    # 1. State creation
    lines.append(
        f'    // Create initial |0...0> state ({n_qubits} qubits, dim={hilbert_dim})'
    )
    lines.append(
        f'    %state = cudm.state_create {{ n_qubits = {n_qubits} : i64 }} '
        f': !cudm.state<{hilbert_dim}>')

    # 2. Operator creation
    lines.append(
        f'    %op = cudm.operator_create {{ n_qubits = {n_qubits} : i64 }} '
        f': !cudm.operator<{hilbert_dim}>')

    # 3. Hamiltonian terms
    for i, term in enumerate(op_program.hamiltonian_terms):
        kind = term.kind
        qubits = term.qubit_indices
        coeff_re = term.coefficient.real
        coeff_im = term.coefficient.imag
        time_dep = "true" if term.time_dependent else "false"
        cb_attr = ""
        if term.callback_id:
            cb_attr = f', callback = "{term.callback_id}"'

        qubit_str = ", ".join(str(q) for q in qubits)
        lines.append(f'    cudm.operator_add_term %op '
                     f'{{ kind = "{kind}", qubits = [{qubit_str}], '
                     f'coeff_re = {coeff_re:.15e} : f64, '
                     f'coeff_im = {coeff_im:.15e} : f64, '
                     f'time_dependent = {time_dep}{cb_attr} }} '
                     f': !cudm.operator<{hilbert_dim}>')

    # 4. Lindblad collapse operators
    for i, term in enumerate(op_program.dissipator_terms):
        kind = term.kind
        qubits = term.qubit_indices
        coeff_re = term.coefficient.real
        coeff_im = term.coefficient.imag

        qubit_str = ", ".join(str(q) for q in qubits)
        lines.append(f'    cudm.lindblad_add_collapse %op '
                     f'{{ kind = "{kind}", qubits = [{qubit_str}], '
                     f'coeff_re = {coeff_re:.15e} : f64, '
                     f'coeff_im = {coeff_im:.15e} : f64 }} '
                     f': !cudm.operator<{hilbert_dim}>')

    # 5. Time-dependent callbacks (waveform envelopes)
    callbacks = [t for t in op_program.hamiltonian_terms if t.callback_id]
    for cb in callbacks:
        lines.append(f'    cudm.callback @{cb.callback_id.lstrip("@")} '
                     f'{{ kind = "envelope" }} : () -> complex<f64>')

    # 6. Evolve
    lines.append(f'    %result = cudm.evolve %state, %op '
                 f'{{ t_start = {t_start:.6e} : f64, '
                 f't_end = {t_end:.6e} : f64, '
                 f'num_steps = {num_steps} : i64, '
                 f'integrator = "{integrator}" }} '
                 f': !cudm.state<{hilbert_dim}>')

    # 7. Extract state data
    lines.append(
        f'    %data = cudm.state_get_data %result '
        f': !cudm.state<{hilbert_dim}> -> memref<{hilbert_dim}xcomplex<f64>>')
    lines.append(f'    return %data : memref<{hilbert_dim}xcomplex<f64>>')
    lines.append('  }')
    lines.append('}')

    return "\n".join(lines) + "\n"

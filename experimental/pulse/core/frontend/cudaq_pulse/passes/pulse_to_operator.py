# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulse-to-operator lowering pass.

**Reference implementation** -- the production path uses MLIR lowering
via ``--pulse-to-qop`` (see ``core/mlir/conversions/PulseToQOp/PulseToQOp.cpp``).
This Python implementation is kept for testing, debugging, and as
documentation of the lowering semantics. It is used when the env var
``CUDAQ_PULSE_LEGACY_PYTHON_PATH=1`` is set.

Lowers drive ops to time-dependent control terms in the qop (quantum operator)
dialect. Collects per-line static Hamiltonians and emits dissipator ops when
calibration data is attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    duration_of,
    line_id_of,
    tone_id_of,
)

# ---------------------------------------------------------------------------
# Operator program structure
# ---------------------------------------------------------------------------


@dataclass()
class OperatorTerm:
    """A single term in the system Hamiltonian or Lindbladian."""

    kind: str
    qubit_indices: tuple
    coefficient: complex = 1.0 + 0j
    time_dependent: bool = False
    callback_id: str = ""


@dataclass()
class OperatorProgram:
    """The result of pulse-to-operator lowering."""

    name: str = "operator"
    ops: list = field(default_factory=list)
    hamiltonian_terms: list = field(default_factory=list)
    dissipator_terms: list = field(default_factory=list)
    n_qubits: int = 0
    total_time_ns: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_QOP_HANDLER = ValueType.TONE
_QOP_SCALAR = ValueType.WAVEFORM
_QOP_PRODUCT = ValueType.DRIVE_LINE
_QOP_OP = ValueType.READOUT_LINE
_QOP_SUPEROP = ValueType.IQ_DATA


def _build_vid_to_qubit_map(program: Program) -> dict[int, int]:
    """Build a map from every line-type VID to its qubit index.

    Follows the SSA chain: ALLOC produces the initial VID, then each
    DRIVE/WAIT/etc produces a new VID that inherits the same qubit.
    """
    vid_to_qubit: dict[int, int] = {}
    for op in program.ops:
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            q = op.attrs.get("qubit")
            if q is not None:
                for v in op.results:
                    if v.vtype in (ValueType.DRIVE_LINE,
                                   ValueType.READOUT_LINE):
                        vid_to_qubit[v.vid] = int(q)

    changed = True
    while changed:
        changed = False
        for op in program.ops:
            for v_in in op.operands:
                if v_in.vid in vid_to_qubit:
                    q = vid_to_qubit[v_in.vid]
                    for v_out in op.results:
                        if (v_out.vtype in (ValueType.DRIVE_LINE,
                                            ValueType.READOUT_LINE) and
                                v_out.vid not in vid_to_qubit):
                            vid_to_qubit[v_out.vid] = q
                            changed = True

    return vid_to_qubit


def _qubit_index_from_line(op: Op,
                           program: Program,
                           vid_to_qubit: dict[int, int] | None = None) -> int:
    """Extract qubit index from a drive op's line or attrs."""
    qubit_idx = op.attrs.get("qubit_index", op.attrs.get("qubit"))
    if qubit_idx is not None:
        return int(qubit_idx)

    if vid_to_qubit is None:
        vid_to_qubit = _build_vid_to_qubit_map(program)

    lid = line_id_of(op)
    if lid is not None and lid in vid_to_qubit:
        return vid_to_qubit[lid]

    for v in op.operands:
        if v.vid in vid_to_qubit:
            return vid_to_qubit[v.vid]

    if lid is not None:
        raise ValueError(
            f"cannot determine qubit index for drive on line %{lid}; "
            f"add 'qubit' attr to the ALLOC op or the drive op")
    raise ValueError("drive op has no line operand and no qubit attr; "
                     "cannot determine target qubit for Hamiltonian lowering")


def _make_value(vid_counter: list, vtype: ValueType, name: str = "") -> Value:
    """Allocate a fresh Value for the operator program."""
    v = Value(vid=vid_counter[0], vtype=vtype, name=name)
    vid_counter[0] += 1
    return v


# ---------------------------------------------------------------------------
# Static Hamiltonian emission
# ---------------------------------------------------------------------------


def _emit_static_hamiltonian(
    program: Program,
    vid_counter: list,
) -> tuple:
    """Emit static Hamiltonian terms (qubit frequencies as sigma_z)."""
    ops: list[Op] = []
    terms: list[OperatorTerm] = []
    ham_products: list[Value] = []

    for qidx, freq_hz in sorted(program.qubit_freq_hz.items()):
        handler = _make_value(vid_counter, _QOP_HANDLER, f"sz_q{qidx}")
        ops.append(
            Op(
                kind=OpKind.QOP_SPIN,
                operands=(),
                results=(handler,),
                attrs={
                    "target": qidx,
                    "kind": "spin_z"
                },
            ))

        coeff_val = freq_hz * 2.0 * math.pi * 1e-9
        coeff = _make_value(vid_counter, _QOP_SCALAR, f"freq_q{qidx}")
        ops.append(
            Op(
                kind=OpKind.QOP_CONST_SCALAR,
                operands=(),
                results=(coeff,),
                attrs={
                    "real": coeff_val,
                    "imag": 0.0
                },
            ))

        product = _make_value(vid_counter, _QOP_PRODUCT, f"H0_q{qidx}")
        ops.append(
            Op(
                kind=OpKind.QOP_MAKE_PRODUCT,
                operands=(coeff, handler),
                results=(product,),
                attrs={},
            ))
        ham_products.append(product)

        terms.append(
            OperatorTerm(
                kind="static_z",
                qubit_indices=(qidx,),
                coefficient=complex(coeff_val, 0),
                time_dependent=False,
            ))

    return ops, terms, ham_products


# ---------------------------------------------------------------------------
# Drive-to-control lowering
# ---------------------------------------------------------------------------


def _emit_drive_control(
    op: Op,
    drive_idx: int,
    program: Program,
    vid_counter: list,
    vid_to_qubit: dict[int, int] | None = None,
) -> tuple:
    """Lower a drive op to time-dependent control terms (X and optionally Y)."""
    emitted: list[Op] = []
    products: list[Value] = []
    terms: list[OperatorTerm] = []
    qidx = _qubit_index_from_line(op, program, vid_to_qubit=vid_to_qubit)

    cr_target = op.attrs.get("cr_target")
    target_qubit = int(cr_target) if cr_target is not None else qidx

    amplitude = float(op.attrs.get("amplitude", 1.0))
    phase = float(op.attrs.get("phase", 0.0))
    wf_type = op.attrs.get("waveform_type", "square")
    callback_id = f"@drive_envelope_{drive_idx}"

    # X-component (in-phase)
    sx = _make_value(vid_counter, _QOP_HANDLER, f"sx_drive{drive_idx}")
    emitted.append(
        Op(
            kind=OpKind.QOP_SPIN,
            operands=(),
            results=(sx,),
            attrs={
                "target": target_qubit,
                "kind": "spin_x"
            },
        ))

    coeff_x = _make_value(vid_counter, _QOP_SCALAR, f"coeff_x_drive{drive_idx}")
    emitted.append(
        Op(
            kind=OpKind.QOP_CALLBACK_SCALAR,
            operands=(),
            results=(coeff_x,),
            attrs={
                "callback": callback_id,
                "waveform_type": wf_type,
                "quadrature": "I",
            },
        ))

    prod_x = _make_value(vid_counter, _QOP_PRODUCT, f"Hd_x_{drive_idx}")
    emitted.append(
        Op(
            kind=OpKind.QOP_MAKE_PRODUCT,
            operands=(coeff_x, sx),
            results=(prod_x,),
            attrs={},
        ))
    products.append(prod_x)
    terms.append(
        OperatorTerm(
            kind="drive_control_x",
            qubit_indices=(target_qubit,),
            coefficient=complex(amplitude * math.cos(phase), 0),
            time_dependent=True,
            callback_id=callback_id,
        ))

    # Y-component (quadrature) — emitted for DRAG or when phase != 0
    if wf_type == "drag" or abs(math.sin(phase)) > 1e-12:
        sy = _make_value(vid_counter, _QOP_HANDLER, f"sy_drive{drive_idx}")
        emitted.append(
            Op(
                kind=OpKind.QOP_SPIN,
                operands=(),
                results=(sy,),
                attrs={
                    "target": target_qubit,
                    "kind": "spin_y"
                },
            ))

        coeff_y = _make_value(vid_counter, _QOP_SCALAR,
                              f"coeff_y_drive{drive_idx}")
        emitted.append(
            Op(
                kind=OpKind.QOP_CALLBACK_SCALAR,
                operands=(),
                results=(coeff_y,),
                attrs={
                    "callback": f"{callback_id}_Q",
                    "waveform_type": wf_type,
                    "quadrature": "Q",
                },
            ))

        prod_y = _make_value(vid_counter, _QOP_PRODUCT, f"Hd_y_{drive_idx}")
        emitted.append(
            Op(
                kind=OpKind.QOP_MAKE_PRODUCT,
                operands=(coeff_y, sy),
                results=(prod_y,),
                attrs={},
            ))
        products.append(prod_y)
        terms.append(
            OperatorTerm(
                kind="drive_control_y",
                qubit_indices=(target_qubit,),
                coefficient=complex(0, amplitude * math.sin(phase)),
                time_dependent=True,
                callback_id=f"{callback_id}_Q",
            ))

    return emitted, terms, products


# ---------------------------------------------------------------------------
# Dissipator emission
# ---------------------------------------------------------------------------


def _emit_dissipators(
    program: Program,
    vid_counter: list,
    t1_times: dict = None,
    t2_times: dict = None,
) -> tuple:
    """Emit Lindblad dissipator ops from calibration data."""
    ops: list[Op] = []
    terms: list[OperatorTerm] = []
    t1_times = t1_times or {}
    t2_times = t2_times or {}

    for qidx in sorted(program.qubit_freq_hz.keys()):
        t1 = t1_times.get(qidx)
        if t1 is not None and t1 > 0:
            gamma1 = 1.0 / t1
            sm = _make_value(vid_counter, _QOP_HANDLER, f"sm_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_SPIN,
                    operands=(),
                    results=(sm,),
                    attrs={
                        "target": qidx,
                        "kind": "lowering"
                    },
                ))
            gamma_coeff = _make_value(vid_counter, _QOP_SCALAR,
                                      f"gamma1_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_CONST_SCALAR,
                    operands=(),
                    results=(gamma_coeff,),
                    attrs={
                        "real": gamma1**0.5,
                        "imag": 0.0
                    },
                ))
            L = _make_value(vid_counter, _QOP_PRODUCT, f"L1_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_MAKE_PRODUCT,
                    operands=(gamma_coeff, sm),
                    results=(L,),
                    attrs={},
                ))
            terms.append(
                OperatorTerm(
                    kind="dissipator_t1",
                    qubit_indices=(qidx,),
                    coefficient=complex(gamma1**0.5, 0),
                ))

        t2 = t2_times.get(qidx)
        if t2 is not None and t2 > 0:
            gamma_phi = 1.0 / t2
            if t1 is not None and t1 > 0:
                gamma_phi = max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
            gamma2 = gamma_phi
            sz = _make_value(vid_counter, _QOP_HANDLER, f"sz_deph_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_SPIN,
                    operands=(),
                    results=(sz,),
                    attrs={
                        "target": qidx,
                        "kind": "spin_z"
                    },
                ))
            gamma_coeff = _make_value(vid_counter, _QOP_SCALAR,
                                      f"gamma2_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_CONST_SCALAR,
                    operands=(),
                    results=(gamma_coeff,),
                    attrs={
                        "real": gamma2**0.5,
                        "imag": 0.0
                    },
                ))
            L = _make_value(vid_counter, _QOP_PRODUCT, f"L2_q{qidx}")
            ops.append(
                Op(
                    kind=OpKind.QOP_MAKE_PRODUCT,
                    operands=(gamma_coeff, sz),
                    results=(L,),
                    attrs={},
                ))
            terms.append(
                OperatorTerm(
                    kind="dissipator_t2",
                    qubit_indices=(qidx,),
                    coefficient=complex(gamma2**0.5, 0),
                ))

    return ops, terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_loop_aware_time(program: Program) -> float:
    """Compute total time accounting for loop iteration counts."""
    total = 0.0
    multiplier_stack: list[int] = []
    current_mul = 1

    for op in program.ops:
        if op.kind == OpKind.FOR_LOOP:
            count = int(op.attrs.get("ub", op.attrs.get("count", 1)))
            multiplier_stack.append(current_mul)
            current_mul *= count
        elif op.kind == OpKind.END_FOR:
            if multiplier_stack:
                current_mul = multiplier_stack.pop()
        else:
            total += duration_of(op) * current_mul

    return total * program.vtu_to_ns


def run_pulse_to_operator(
    program: Program,
    *,
    target: Any = None,
    t1_times: dict = None,
    t2_times: dict = None,
) -> OperatorProgram:
    """Lower a pulse program to an operator program in the qop dialect.

    For each drive op, emit a time-dependent control term. Collect per-line
    static Hamiltonians. Emit dissipator ops if calibration data is present.

    Parameters
    ----------
    program : Program
        The pulse IR program.
    target : Target, optional
        If provided, Hamiltonian and dissipator terms are sourced from
        the target instead of raw dicts.
    t1_times, t2_times : dict, optional
        Legacy per-qubit decoherence dicts. Ignored if ``target`` is provided.
    """
    if target is not None:
        t1_times = {idx: t * 1e-6 for idx, t in target.t1_times.items()}
        t2_times = {idx: t * 1e-6 for idx, t in target.t2_times.items()}
        if not program.qubit_freq_hz:
            program.qubit_freq_hz = dict(target.frequencies)

    vid_counter = [max((v.vid for v in program.values), default=0) + 1000]
    all_ops: list[Op] = []
    all_ham_terms: list[OperatorTerm] = []
    all_diss_terms: list[OperatorTerm] = []

    # Static Hamiltonian
    static_ops, static_terms, ham_products = _emit_static_hamiltonian(
        program, vid_counter)
    all_ops.extend(static_ops)
    all_ham_terms.extend(static_terms)

    # If target provided, also emit anharmonicity and coupling terms
    if target is not None:
        for tdict in target.hamiltonian_terms():
            if tdict["kind"] in ("anharmonicity", "coupling_xx",
                                 "crosstalk_zz"):
                all_ham_terms.append(
                    OperatorTerm(
                        kind=tdict["kind"],
                        qubit_indices=tdict["qubit_indices"],
                        coefficient=tdict["coefficient"],
                        time_dependent=tdict.get("time_dependent", False),
                    ))

    # Drive control terms
    vid_to_qubit = _build_vid_to_qubit_map(program)
    drive_idx = 0
    for op in program.ops:
        if op.kind == OpKind.DRIVE:
            drive_ops, drive_terms, drive_products = _emit_drive_control(
                op, drive_idx, program, vid_counter, vid_to_qubit=vid_to_qubit)
            all_ops.extend(drive_ops)
            all_ham_terms.extend(drive_terms)
            ham_products.extend(drive_products)
            drive_idx += 1

    # Sum all Hamiltonian terms
    if len(ham_products) > 1:
        H_total = _make_value(vid_counter, _QOP_OP, "H_total")
        all_ops.append(
            Op(
                kind=OpKind.QOP_MAKE_SUM,
                operands=tuple(ham_products),
                results=(H_total,),
                attrs={},
            ))

    # Dissipators
    diss_ops, diss_terms = _emit_dissipators(program, vid_counter, t1_times,
                                             t2_times)
    all_ops.extend(diss_ops)
    all_diss_terms.extend(diss_terms)

    # If target provided, also use target's dissipator terms
    if target is not None:
        for tdict in target.dissipator_terms():
            all_diss_terms.append(
                OperatorTerm(
                    kind=tdict["kind"],
                    qubit_indices=tdict["qubit_indices"],
                    coefficient=tdict["coefficient"],
                ))

    # Lindblad superoperator if dissipators present
    if (diss_ops or all_diss_terms) and ham_products:
        lindblad_val = _make_value(vid_counter, _QOP_SUPEROP, "lindbladian")
        all_ops.append(
            Op(
                kind=OpKind.QOP_LINDBLAD,
                operands=(),
                results=(lindblad_val,),
                attrs={"n_collapse_ops": len(all_diss_terms)},
            ))

    n_qubits = len(program.qubit_freq_hz) if program.qubit_freq_hz else 1
    total_time = _compute_loop_aware_time(program)

    return OperatorProgram(
        name=f"{program.name}_operator",
        ops=all_ops,
        hamiltonian_terms=all_ham_terms,
        dissipator_terms=all_diss_terms,
        n_qubits=n_qubits,
        total_time_ns=total_time,
    )

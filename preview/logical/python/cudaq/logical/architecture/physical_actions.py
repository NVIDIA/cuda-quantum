# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable physical instruction definitions.

These are ordinary Python values.  A device advertises the subset it supports
through ``ResourceClass.native_actions``; no process-global action catalog is
queried during compilation.
"""

from __future__ import annotations

from cudaq.logical.architecture.physical_definition import (
    QuantumProcess,
    NativeActionDecomposition,
    NativeActionStep,
    PhysicalAction,
)

H = PhysicalAction(
    "h",
    arity=1,
    process=QuantumProcess("h"),
    controller_bindings={"lanes": "h"},
)
S = PhysicalAction(
    "s",
    arity=1,
    process=QuantumProcess("s"),
    controller_bindings={"lanes": "s"},
)
SDG = PhysicalAction(
    "sdg",
    arity=1,
    process=QuantumProcess("sdg"),
    controller_bindings={"lanes": "sdg"},
)
T = PhysicalAction(
    "t",
    arity=1,
    process=QuantumProcess("t"),
    controller_bindings={"lanes": "t"},
)
TDG = PhysicalAction(
    "tdg",
    arity=1,
    process=QuantumProcess("tdg"),
    controller_bindings={"lanes": "tdg"},
)
X = PhysicalAction(
    "x",
    arity=1,
    process=QuantumProcess("x"),
    controller_bindings={"lanes": "x"},
)
Y = PhysicalAction(
    "y",
    arity=1,
    process=QuantumProcess("y"),
    controller_bindings={"lanes": "y"},
)
Z = PhysicalAction(
    "z",
    arity=1,
    process=QuantumProcess("z"),
    controller_bindings={"lanes": "z"},
)
RESET = PhysicalAction(
    "reset",
    arity=1,
    process=QuantumProcess("reset"),
    controller_bindings={"lanes": "reset"},
)
CX = PhysicalAction(
    "cx",
    arity=2,
    process=QuantumProcess("cx"),
    controller_bindings={"lanes": "cx"},
)
CZ = PhysicalAction(
    "cz",
    arity=2,
    process=QuantumProcess("cz"),
    controller_bindings={"lanes": "cz"},
)
SWAP = PhysicalAction(
    "swap",
    arity=2,
    process=QuantumProcess("swap"),
    controller_bindings={"lanes": "swap"},
)

# The same logical CZ semantics is realized by a Rydberg-blockade physical
# event. Its distinct identity preserves calibration while target projections
# remain explicit.
RYDBERG_CZ = PhysicalAction(
    "rydberg_cz",
    arity=2,
    process=QuantumProcess("cz"),
    controller_bindings={"lanes": "cz"},
)

GLOBAL_H = PhysicalAction(
    "global_h",
    arity=1,
    broadcast=True,
    process=QuantumProcess("h"),
    controller_bindings={"lanes": "h"},
)
GLOBAL_X = PhysicalAction(
    "global_x",
    arity=1,
    broadcast=True,
    process=QuantumProcess("x"),
    controller_bindings={"lanes": "x"},
)

# Trapped-ion native vocabulary. The parameterized rotations and the ZZ
# parity interaction deliberately carry NO fixed target gate semantics: a
# Clifford spelling would only be honest at one angle, and a target must
# fail closed rather than infer behavior from a name. Movement actions are
# transport events whose legality comes from zone topology.
ZZ = PhysicalAction(
    "zz",
    arity=2,
    process=QuantumProcess("zz_phase"),
    parameters=("angle",),
)
RZ = PhysicalAction(
    "rz",
    arity=1,
    process=QuantumProcess("rz"),
    parameters=("angle",),
)
PHASED_X = PhysicalAction(
    "phased_x",
    arity=1,
    process=QuantumProcess("phased_x"),
    parameters=("angle", "phase"),
)
ION_SPLIT = PhysicalAction("split",
                           arity=1,
                           process=QuantumProcess("transport_split"))
ION_MERGE = PhysicalAction("merge",
                           arity=1,
                           process=QuantumProcess("transport_merge"))
ION_TRANSPORT = PhysicalAction("transport",
                               arity=1,
                               process=QuantumProcess("transport"))


def clifford_set() -> tuple[PhysicalAction, ...]:
    return (H, S, SDG, X, Y, Z, CX, CZ, SWAP, RESET)


def qccd_native_set(
        *,
        one_qubit=("Rz", "PhasedX"),
        two_qubit=("ZZPhase",),
        movement=("split", "merge", "transport"),
) -> tuple[PhysicalAction, ...]:
    """The trapped-ion QCCD vocabulary: rotations, parity gates, movement."""

    by_name = {
        "Rz": RZ,
        "PhasedX": PHASED_X,
        "ZZPhase": ZZ,
        "ZZ": ZZ,
        "split": ION_SPLIT,
        "merge": ION_MERGE,
        "transport": ION_TRANSPORT,
    }
    selected = []
    for name in (*one_qubit, *two_qubit, *movement):
        action = by_name.get(str(name))
        if action is None:
            raise ValueError(f"unknown QCCD native action {name!r}")
        selected.append(action)
    return (*selected, RESET)


def superconducting_clifford_set() -> tuple[PhysicalAction, ...]:
    return clifford_set()


def cz_native_clifford_set() -> tuple[PhysicalAction, ...]:
    """A CZ-native transmon vocabulary: single-qubit Cliffords + ``CZ``, no ``CX``.

    Real superconducting hardware (e.g. Google Sycamore) entangles only through
    ``CZ``; a ``CX`` must be realized as ``H(t); CZ(c, t); H(t)``. Advertising a
    native set without ``cx`` makes the physical projection perform that
    decomposition, so the emitted circuit carries the single-qubit rotations the
    machine really runs.
    """

    return (H, S, SDG, X, Y, Z, CZ, RESET)


def neutral_atom_set() -> tuple[PhysicalAction, ...]:
    return (H, S, SDG, X, Y, Z, GLOBAL_H, GLOBAL_X, RYDBERG_CZ, RESET)


def neutral_atom_decompositions() -> tuple[NativeActionDecomposition, ...]:
    """Legalize common semantic Clifford actions to blockade-native pulses."""

    cx_steps = (
        NativeActionStep(H, 1),
        NativeActionStep(RYDBERG_CZ, 0, 1),
        NativeActionStep(H, 1),
    )
    return (
        NativeActionDecomposition(
            CZ,
            (NativeActionStep(RYDBERG_CZ, 0, 1),),
        ),
        NativeActionDecomposition(CX, cx_steps),
        NativeActionDecomposition(
            SWAP,
            (
                *cx_steps,
                NativeActionStep(H, 0),
                NativeActionStep(RYDBERG_CZ, 1, 0),
                NativeActionStep(H, 0),
                *cx_steps,
            ),
        ),
    )


__all__ = [
    "PhysicalAction",
    "H",
    "S",
    "SDG",
    "T",
    "TDG",
    "X",
    "Y",
    "Z",
    "RESET",
    "CX",
    "CZ",
    "SWAP",
    "RYDBERG_CZ",
    "GLOBAL_H",
    "GLOBAL_X",
    "clifford_set",
    "superconducting_clifford_set",
    "cz_native_clifford_set",
    "qccd_native_set",
    "ZZ",
    "RZ",
    "PHASED_X",
    "neutral_atom_set",
    "neutral_atom_decompositions",
]

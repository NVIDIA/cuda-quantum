# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Composable reversible helpers over the existing P0 logical actions in CUDA-Q Logical.

The helpers in this module are ordinary Python authoring routines.  They emit
only ``x``, ``h``, ``cx``, and ``ccz`` operations into the active trace; no
helper is a new logical or machine primitive.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from ..errors import (
    CrossContextValue,
    InvalidReversibleCall,
    InvalidReversibleSignature,
)
from ..ops._impl import ccz as _ccz
from ..ops._impl import cx as _cx
from ..ops._impl import h as _h
from ..ops._impl import x as _x
from ..programs.context import current_trace
from ..types.values import UseAfterConsume, logical_qubit


class MCXPolicy(str, Enum):
    """Supported multi-controlled-X decomposition policies."""

    CLEAN_LADDER = "clean_ladder"


def _preflight_handles(
    operation: str,
    named_values: Iterable[tuple[str, object]],
) -> None:
    """Validate one complete helper boundary before any operation is emitted."""

    values = tuple(named_values)
    for name, value in values:
        if not isinstance(value, logical_qubit):
            raise InvalidReversibleSignature(
                f"{operation} {name} must be a cudaq.logical.types.logical_qubit"
            )
    trace = current_trace()
    canonical_values: list[tuple[str, logical_qubit]] = []
    resolver = (None if trace is None else getattr(
        trace, "_canonical_qubit_for_ssa", None))
    if resolver is not None:
        expected_type = str(trace.logical_type)
        for name, value in values:
            try:
                actual_type = str(value.type)
            except (AttributeError, TypeError, ValueError) as error:
                raise InvalidReversibleSignature(
                    f"{operation} {name} does not carry a valid CUDA-Q Logical SSA value"
                ) from error
            if actual_type != expected_type:
                raise InvalidReversibleSignature(
                    f"{operation} {name} must carry {expected_type}; "
                    f"received {actual_type}")
            canonical = resolver(value.mlir_value)
            if canonical is None:
                raise CrossContextValue(
                    f"{operation} {name} SSA value was not minted by the "
                    "active CUDA-Q Logical builder/domain")
            canonical_values.append((name, canonical))

        for name, canonical in canonical_values:
            if canonical.owner is not trace:
                raise CrossContextValue(
                    f"{operation} {name} canonical owner metadata does not "
                    "match the active CUDA-Q Logical builder/domain")
            if not canonical.is_live:
                raise UseAfterConsume(
                    f"{operation} {name} is not a live logical-qubit owner")
    else:
        # Outside a real P0 builder, retain the ordinary signature/liveness and
        # pairwise-domain checks.  The first primitive remains responsible for
        # reporting NoActiveTrace for otherwise-valid input.
        canonical_values = list(values)
        for name, value in canonical_values:
            if not value.is_live:
                raise UseAfterConsume(
                    f"{operation} {name} is not a live logical-qubit owner")
        owner = canonical_values[0][1].owner if trace is None else trace
        for name, value in canonical_values:
            if value.owner is not owner:
                raise CrossContextValue(
                    f"{operation} {name} belongs to a different CUDA-Q Logical "
                    "builder/domain")

    seen: list[tuple[str, object]] = []
    for name, canonical in canonical_values:
        for previous_name, previous_ssa in seen:
            try:
                aliases = canonical.mlir_value == previous_ssa
            except (TypeError, ValueError):
                aliases = False
            if aliases:
                raise InvalidReversibleCall(
                    f"{operation} requires distinct logical-qubit SSA values; "
                    f"{name} aliases {previous_name}")
        seen.append((name, canonical.mlir_value))

    if resolver is not None:
        for (name, value), (_, canonical) in zip(values, canonical_values):
            if value is not canonical:
                raise InvalidReversibleCall(
                    f"{operation} {name} is not the canonical live handle "
                    "minted for its SSA value by the active CUDA-Q Logical builder"
                )


def _emit_ccx(
    control_a: logical_qubit,
    control_b: logical_qubit,
    target: logical_qubit,
) -> tuple[logical_qubit, logical_qubit, logical_qubit]:
    target = _h(target)
    control_a, control_b, target = _ccz(control_a, control_b, target)
    target = _h(target)
    return control_a, control_b, target


def ccx(
    control_a: logical_qubit,
    control_b: logical_qubit,
    target: logical_qubit,
) -> tuple[logical_qubit, logical_qubit, logical_qubit]:
    """Apply positive-control Toffoli as ``H(target); CCZ; H(target)``.

    The three inputs are consumed and three fresh SSA handles are returned in
    the same order.
    """

    _preflight_handles(
        "ccx",
        (
            ("control_a", control_a),
            ("control_b", control_b),
            ("target", target),
        ),
    )
    return _emit_ccx(control_a, control_b, target)


def cswap(
    control: logical_qubit,
    left: logical_qubit,
    right: logical_qubit,
) -> tuple[logical_qubit, logical_qubit, logical_qubit]:
    """Apply positive-control Fredkin using two CX gates and one CCX.

    The decomposition is ``CX(left, right); CCX(control, right, left);``
    ``CX(left, right)``.  Every input owner is returned as a fresh SSA handle.
    """

    _preflight_handles(
        "cswap",
        (("control", control), ("left", left), ("right", right)),
    )
    left, right = _cx(left, right)
    control, right, left = _emit_ccx(control, right, left)
    left, right = _cx(left, right)
    return control, left, right


def mcx(
    controls: Iterable[logical_qubit],
    target: logical_qubit,
    *,
    ancillas: Iterable[logical_qubit] = (),
    policy: MCXPolicy,
) -> tuple[tuple[logical_qubit, ...], logical_qubit, tuple[logical_qubit, ...]]:
    """Apply a positive-control multi-controlled X.

    ``policy`` is required and currently accepts only
    :attr:`MCXPolicy.CLEAN_LADDER`.  For more than two controls, callers must
    supply exactly ``len(controls) - 2`` distinct ancillas known by the caller
    to be in ``|0>``.  CUDA-Q Logical logical handles do not prove a quantum basis state,
    so cleanliness is an explicit precondition rather than a trace-time check.
    Conditional on that precondition, every ancilla is restored to ``|0>``.

    The result is ``(controls, target, ancillas)`` with both collections
    normalized to tuples of fresh SSA handles.  Boundary cases use X for zero
    controls, CX for one, and CCX for two.
    """

    if not isinstance(policy, MCXPolicy):
        raise InvalidReversibleSignature(
            "mcx policy= must be a cudaq.logical.algorithms.reversible.MCXPolicy"
        )
    if policy is not MCXPolicy.CLEAN_LADDER:
        raise InvalidReversibleCall(f"unsupported mcx policy {policy!r}")

    try:
        controls = tuple(controls)
    except TypeError as error:
        raise InvalidReversibleSignature(
            "mcx controls must be a finite iterable of logical qubits"
        ) from error
    try:
        ancillas = tuple(ancillas)
    except TypeError as error:
        raise InvalidReversibleSignature(
            "mcx ancillas must be a finite iterable of logical qubits"
        ) from error

    required_ancillas = max(0, len(controls) - 2)
    if len(ancillas) != required_ancillas:
        raise InvalidReversibleCall(
            f"mcx CLEAN_LADDER with {len(controls)} controls requires exactly "
            f"{required_ancillas} clean ancilla(s); received {len(ancillas)}")

    _preflight_handles(
        "mcx",
        (
            *((f"controls[{index}]", value)
              for index, value in enumerate(controls)),
            ("target", target),
            *((f"ancillas[{index}]", value)
              for index, value in enumerate(ancillas)),
        ),
    )

    if not controls:
        return controls, _x(target), ancillas
    if len(controls) == 1:
        controls = list(controls)
        controls[0], target = _cx(controls[0], target)
        return tuple(controls), target, ancillas
    if len(controls) == 2:
        control_a, control_b, target = _emit_ccx(*controls, target)
        return (control_a, control_b), target, ancillas

    controls = list(controls)
    ancillas = list(ancillas)
    controls[0], controls[1], ancillas[0] = _emit_ccx(controls[0], controls[1],
                                                      ancillas[0])
    for index in range(2, len(controls) - 1):
        controls[index], ancillas[index - 2], ancillas[index - 1] = _emit_ccx(
            controls[index], ancillas[index - 2], ancillas[index - 1])
    controls[-1], ancillas[-1], target = _emit_ccx(controls[-1], ancillas[-1],
                                                   target)
    for index in range(len(controls) - 2, 1, -1):
        controls[index], ancillas[index - 2], ancillas[index - 1] = _emit_ccx(
            controls[index], ancillas[index - 2], ancillas[index - 1])
    controls[0], controls[1], ancillas[0] = _emit_ccx(controls[0], controls[1],
                                                      ancillas[0])
    return tuple(controls), target, tuple(ancillas)


__all__ = ["MCXPolicy", "ccx", "cswap", "mcx"]

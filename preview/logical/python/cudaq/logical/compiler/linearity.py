# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Linear-use verification over canonical QLX / LVM / Fabric / Phys bodies.

Logical qubits, encoded patches, consumable resource states, linear events,
and value-semantic frames are *linear*: ordinary MLIR SSA permits any number
of uses, but every linear SSA value must have exactly one owner along every
execution path.

The rule lives in C++ (``cudaq/Optimizer/Analysis/LinearOwnership.h``) and is
driven by the IR itself rather than by names:

- A type is linear when it implements ``cudaq::LinearTypeInterface``.
  ``mayDrop()`` types (frames) may go unconsumed.
- An op with the ``LinearBorrowsOperands`` trait observes its operands;
  every other use consumes.
- ``LinearExclusiveRegions`` ops (``cflow.if``, ``event.try_take``) run one
  region, so one consume per region is one dynamic consume, and a captured
  value must be consumed in every region.
- ``LinearRepeatedRegions`` ops (``cflow.repeat``, ``cflow.while``) may not
  consume a value defined outside them.
- ``LinearOwnershipScope`` ops (``qlx.program``, ``qlx.objective_body``,
  ``lvm.kernel``, ``fabric.gadget``/``protocol``/``circuit``,
  ``phys.graph``) are checked independently.

The ``qlx.program`` and ``qlx.objective_body`` verifiers run the same
analysis, so P0 that breaks the rule does not verify. ``qlx-opt
--verify-linear-ownership`` (a CUDA-Q core pass) runs it over every tier. This module calls that
analysis and wraps its result for evidence records and error reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .._native import native


def is_linear_type(mlir_type) -> bool:
    """True when values of this type are single-owner linear values."""
    return native.type_is_linear(mlir_type)


@dataclass(frozen=True, slots=True)
class LinearityViolation:
    """One linear-ownership violation at a concrete IR location."""

    body: str
    value: str
    # double-consume | loop-capture-consume | borrow-after-consume | unconsumed
    kind: str
    message: str
    location: str

    def render(self) -> str:
        return f"[{self.kind}] {self.body}: {self.message} at {self.location}"


@dataclass(frozen=True, slots=True)
class LinearityReport:
    """Structured result of one linear-use verification run."""

    checked_bodies: tuple[str, ...]
    linear_values: int
    violations: tuple[LinearityViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def result(self) -> str:
        if not self.checked_bodies:
            return "skipped"
        return "pass" if self.ok else "fail"


class LinearityError(ValueError):
    """Linear single-ownership verification failed."""

    def __init__(self, report: LinearityReport, subject: str | None = None):
        self.report = report
        header = "linear ownership verification failed"
        if subject:
            header += f" for {subject}"
        lines = [header] + [
            "  - " + violation.render() for violation in report.violations
        ]
        super().__init__("\n".join(lines))


def check_linearity(module) -> LinearityReport:
    """Run the linear single-ownership analysis over every body in a module.

    Returns a structured :class:`LinearityReport`; never raises for
    violations. ``module`` is a ``cudaq.mlir.ir.Module`` or an operation.
    Declaration-only bodies (a bare terminator and unused arguments) are
    skipped.
    """
    operation = getattr(module, "operation", module)
    raw = json.loads(native.analyze_linear_ownership_json(operation))
    return LinearityReport(
        checked_bodies=tuple(raw["checked_bodies"]),
        linear_values=int(raw["linear_values"]),
        violations=tuple(
            LinearityViolation(
                body=item["body"],
                value=item["value"],
                kind=item["kind"],
                message=item["message"],
                location=item["location"],
            ) for item in raw["violations"]),
    )


def verify_linearity(module, *, subject: str | None = None) -> LinearityReport:
    """Run :func:`check_linearity` and raise :class:`LinearityError` on any
    violation.  Returns the (passing or vacuous) report otherwise."""
    report = check_linearity(module)
    if not report.ok:
        raise LinearityError(report, subject=subject)
    return report

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed analysis entry points over immutable QLX definitions/builds."""

from __future__ import annotations

from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.gadgets.definition import GadgetDefinition
from cudaq.logical.programs.definition import ProgramDefinition
from ..std import LogicalActionRef


def clifford_action(value) -> CliffordAction:
    """Derive the exact signed symplectic action of logical intent.

    Analysis consumes a logical action definition, or a gadget whose
    ``implements=`` objective is such a definition. It never infers intent
    from the physical body: realization equivalence remains an independent
    verifier obligation.
    """

    if isinstance(value, CliffordAction):
        return value
    if isinstance(value, GadgetDefinition):
        value = value.implements
    if isinstance(value, LogicalActionRef):
        return CliffordAction.standard(value.name, value.arity)
    if isinstance(value, ProgramDefinition):
        return CliffordAction.from_program(value)
    operation = getattr(value, "operation", value)
    if getattr(operation, "name", None) == "qlx.program":
        function_type = operation.attributes["function_type"].value
        return CliffordAction.from_mlir_program(
            operation,
            ports=tuple(range(len(function_type.inputs))),
        )
    raise TypeError(
        "cudaq.logical.analysis.clifford_action expects an action-like "
        "@cudaq.logical.objective, a gadget implementing one, a qlx.program operation, "
        "or CliffordAction")


__all__ = ["clifford_action"]

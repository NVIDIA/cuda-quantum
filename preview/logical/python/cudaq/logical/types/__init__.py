# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical values, references, annotations, and traced proxies."""

from cudaq.logical.types.values import (
    EventState,
    EventStatusValue,
    Float64Value,
    IndexValue,
    LogicalBool,
    LogicalEventValue,
    LogicalFrameValue,
    LogicalRegister,
    LogicalResourceValue,
    logical_qubit,
)
from cudaq.logical.algebra.angle import (
    Angle,
    pi,
)
from cudaq.logical.programs.binding import (
    LogicalPortRef,
    LogicalPorts,
    ObjectiveOperandRef,
    ObjectiveOperands,
)
from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.programs.definition import (
    Definition,
    DefinitionHandle,
    ProgramDefinition,
)
from cudaq.logical.gadgets import patch
from cudaq.logical.algebra.pauli import (
    I,
    X,
    Y,
    Z,
    PauliFactor,
    PauliGroupElement,
    PauliProduct,
)
from cudaq.logical.types.semantic import (
    LogicalState,
    event,
    float64,
    index,
    logical_event,
    logical_frame,
    logical_record,
    logical_resource,
    plus,
    record,
    resource,
    zero,
)
from cudaq.logical.std import FrameDomain, ResourceKind

__all__ = [
    "Angle",
    "pi",
    "FrameDomain",
    "ResourceKind",
    "logical_qubit",
    "LogicalBool",
    "LogicalEventValue",
    "LogicalFrameValue",
    "LogicalResourceValue",
    "LogicalRegister",
    "IndexValue",
    "Float64Value",
    "EventState",
    "EventStatusValue",
    "CliffordAction",
    "ObjectiveOperandRef",
    "ObjectiveOperands",
    "LogicalState",
    "LogicalPortRef",
    "LogicalPorts",
    "Definition",
    "DefinitionHandle",
    "PauliFactor",
    "PauliGroupElement",
    "PauliProduct",
    "ProgramDefinition",
    "I",
    "X",
    "Y",
    "Z",
    "float64",
    "index",
    "logical_event",
    "logical_frame",
    "logical_record",
    "logical_resource",
    "event",
    "record",
    "resource",
    "plus",
    "zero",
]

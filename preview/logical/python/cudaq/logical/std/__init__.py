# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogicalActionRef:
    name: str
    arity: int

    @property
    def operands(self):
        from ..programs.binding import (
            ObjectiveOperands,
            standard_objective_operand_names,
        )

        return ObjectiveOperands(
            self, standard_objective_operand_names(self.name, self.arity))


@dataclass(frozen=True, slots=True)
class LogicalInstrumentRef:
    name: str
    arity: int
    result_arity: int

    @property
    def operands(self):
        from ..programs.binding import (
            ObjectiveOperands,
            standard_objective_operand_names,
        )

        return ObjectiveOperands(
            self, standard_objective_operand_names(self.name, self.arity))


@dataclass(frozen=True, slots=True)
class ResourceFlowRef:
    kind: str
    resource: "ResourceKind"
    source: object = None
    destination: object = None

    @property
    def name(self) -> str:
        suffix = self.resource.name
        return f"{self.kind}_{suffix}"


@dataclass(frozen=True, slots=True)
class ResourceKind:
    name: str
    consume_action: LogicalActionRef | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource kind name must be nonempty")
        if self.consume_action is not None and not isinstance(
                self.consume_action, LogicalActionRef):
            raise TypeError(
                "resource kind consume_action must be a LogicalActionRef or None"
            )


@dataclass(frozen=True, slots=True)
class FrameDomain:
    name: str


class ObjectiveFamily(str):
    """Typed name of one QEC-lowering objective family.

    A plain ``str`` subclass so ``QECLowering(objective_family=...)`` and
    ``ActionSiteHandle`` comparisons keep their existing string contract while
    call sites reference the family through one typed constant.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"ObjectiveFamily({str.__str__(self)!r})"


@dataclass(frozen=True, slots=True)
class SyndromeExtractionObjective(LogicalInstrumentRef):
    """Parameterized ideal objective: check extraction on one code.

    ``cudaq.logical.std.syndrome_extraction(code)`` instances compare equal exactly
    when they name the same code artifact. ``result_arity`` is the typed check-
    record width used by the extraction boundary; those records are QEC
    analysis facts, not Boolean results in the ideal logical objective ABI.
    """

    code: object = None
    family: str = "syndrome_extraction"


h = LogicalActionRef("h", 1)
s = LogicalActionRef("s", 1)
sdg = LogicalActionRef("sdg", 1)
x = LogicalActionRef("x", 1)
y = LogicalActionRef("y", 1)
z = LogicalActionRef("z", 1)
t = LogicalActionRef("t", 1)
tdg = LogicalActionRef("tdg", 1)
idle = LogicalActionRef("idle", 1)
# ``memory`` is the documentation-friendly name of the idle/memory objective
# family; it is the same artifact, so selection treats them identically.
memory = idle
cx = LogicalActionRef("cx", 2)
cz = LogicalActionRef("cz", 2)
ccz = LogicalActionRef("ccz", 3)

prepare_zero = LogicalInstrumentRef("prepare_zero", 0, 1)
prepare_plus = LogicalInstrumentRef("prepare_plus", 0, 1)
prepare_t = LogicalInstrumentRef("prepare_t", 0, 1)
measure_x = LogicalInstrumentRef("measure_x", 1, 1)
measure_z = LogicalInstrumentRef("measure_z", 1, 1)
# ``cudaq.logical.mpp(...)`` is the authoring operation.  This names the same built-in
# instrument on typed realization/selection surfaces such as
# ``@cudaq.logical.protocol(implements=cudaq.logical.std.mpp)``.
mpp = LogicalInstrumentRef("mpp", 2, 1)
# Parameterized Pauli-product rotations use one typed objective name; masks,
# signs, and angles remain exact action-site specialization parameters.
pauli_rotation = LogicalActionRef("pauli_rotation", 2)

# Documentation-friendly names are aliases, not a second objective family.
H, S, SDG, X, Y, Z, T, TDG = h, s, sdg, x, y, z, t, tdg
CX, CZ, CCZ = cx, cz, ccz
PREPARE_ZERO, PREPARE_PLUS, PREPARE_T = prepare_zero, prepare_plus, prepare_t
MEASURE_X, MEASURE_Z = measure_x, measure_z

# Objective-family tags matched by QEC lowerings (see
# ``QECLowering(objective_family=...)`` and the P1 action-site classifier).
pauli_product_measurement = ObjectiveFamily("pauli_product_measurement")
pauli_product_rotation = ObjectiveFamily("pauli_product_rotation")

T_STATE = ResourceKind("t_state", consume_action=t)
Y_STATE = ResourceKind("y_state", consume_action=s)
RAW_T_STATE = ResourceKind("raw_t_state")
CCZ_STATE = ResourceKind("ccz_state", consume_action=ccz)
CS_STATE = ResourceKind("cs_state")
ENCODED_BELL_PAIR = ResourceKind("encoded_bell_pair")
PAULI_FRAME = FrameDomain("pauli_frame")


def produce(resource: ResourceKind, *, code=None) -> ResourceFlowRef:
    if not isinstance(resource, ResourceKind):
        raise TypeError("logical.produce expects a ResourceKind")
    return ResourceFlowRef("produce", resource, destination=code)


def transport(resource: ResourceKind, *, source=None, destination=None):
    if not isinstance(resource, ResourceKind):
        raise TypeError("logical.transport expects a ResourceKind")
    return ResourceFlowRef("transport",
                           resource,
                           source=source,
                           destination=destination)


def syndrome_extraction(code) -> SyndromeExtractionObjective:
    """Typed ideal objective for one round of check extraction on ``code``."""
    from ..codes import (
        Code,
        Encoding,
    )

    if isinstance(code, Encoding):
        code = code.code
    if not isinstance(code, Code):
        raise TypeError(
            "logical.syndrome_extraction expects a cudaq.logical.Code or cudaq.logical.Encoding"
        )
    checks = (code.block.partitions.get("sx", 0) +
              code.block.partitions.get("sz", 0)) or len(code.stabilizers)
    if not checks:
        raise ValueError(f"code {code.name!r} declares no checks to extract")
    return SyndromeExtractionObjective(
        name=f"syndrome_extraction_{code.name}",
        arity=1,
        result_arity=checks,
        code=code,
    )

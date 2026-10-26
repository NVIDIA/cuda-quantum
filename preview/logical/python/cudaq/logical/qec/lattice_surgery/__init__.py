# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed, provider-neutral lattice-surgery compilation.

QLX owns the semantic problem, temporal mapper, immutable provider-bound plan,
and device dispatch contract. A compiler provider owns its microarchitecture,
complete-batch feasibility oracle, and exact P2 protocol realization; physical
projection is a separate device capability. Provider geometry, naming, and
lowering logic remain outside core QLX.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ...compiler._scheduling import SchedulingStrategy, scheduling
from ...errors import PlacementInfeasible
from cudaq.logical.programs.definition import (
    DefinitionHandle,
    ProgramDefinition,
)
from cudaq.logical.devices.definition import (
    Device,
    QECChannelRealization,
    QECChannelPort,
)
from cudaq.logical.codes import (
    Encoding,
    QECBlockOwner,
)
from cudaq.logical.gadgets import GadgetDefinition
from cudaq.logical.algebra.pauli import (
    PauliFactor,
    PauliProduct,
)
from cudaq.logical.stages import (
    P0,
    P1,
    P2,
    P3,
)
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.qec.lowering import (
    QECCompiler,
    QECNetworkCompiler,
    QECNetworkContext,
    QECLowering,
)
from cudaq.logical.architecture.logical import (
    SpaceSlot,
    capability,
)
from ._codec import (
    _array,
    _digest,
    _frozen_mapping,
    _json_value,
    _mapping_module,
    _pipeline_digest,
    _pipeline_record,
    _record,
    _require_commitment_digest,
    _require_digest,
    _strict_json_loads,
)
from ._mpp import (
    AncillaPolicy,
    SurgeryPrimitives,
    YBasisPolicy,
    connected_ancilla,
    local_basis_change,
    mpp_compiler,
    surface_primitives,
)


def _device_spaces(device: Device) -> dict[str, Any]:
    if not isinstance(device, Device):
        raise TypeError(
            "lattice-surgery artifact replay requires a typed cudaq.logical.Device"
        )
    spaces = {value.name: value for value in device.logical.spaces}
    if None in spaces:
        raise ValueError(
            "lattice-surgery artifact replay requires named device spaces")
    return spaces


def _slot_from_record(value, *, device: Device, what: str) -> SpaceSlot:
    value = _record(
        value,
        what=what,
        keys=("space", "index"),
    )
    name = value["space"]
    index = value["index"]
    if not isinstance(name, str) or not name:
        raise TypeError(f"{what} space must be a nonempty string")
    if (not isinstance(index, int) or isinstance(index, bool) or index < 0):
        raise TypeError(f"{what} index must be a nonnegative integer")
    try:
        space = _device_spaces(device)[name]
    except KeyError as exc:
        raise ValueError(
            f"{what} references absent device space @{name}") from exc
    try:
        return space[index]
    except IndexError as exc:
        raise ValueError(
            f"{what} index {index} exceeds @{name} capacity") from exc


def _slot_record(slot: SpaceSlot) -> dict[str, Any]:
    name = slot.space.name
    if not isinstance(name, str) or not name:
        raise ValueError("lattice-surgery slots require named logical spaces")
    if slot.index is None:
        raise ValueError(
            "lattice-surgery compilation requires a concrete logical slot")
    return {
        "space": name,
        "index": slot.index,
    }


def _require_slot(value, *, what: str) -> SpaceSlot:
    if not isinstance(value, SpaceSlot):
        raise TypeError(f"{what} requires a typed cudaq.logical.SpaceSlot")
    _slot_record(value)
    return value


@dataclass(frozen=True, slots=True)
class PauliTerm:
    """One logical Pauli factor bound to a concrete P1 slot."""

    slot: SpaceSlot
    pauli: str

    def __post_init__(self) -> None:
        _require_slot(self.slot, what="lattice-surgery Pauli term")
        pauli = str(self.pauli).upper()
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError("lattice-surgery Pauli terms require X, Y, or Z")
        object.__setattr__(self, "pauli", pauli)

    def to_dict(self) -> dict[str, Any]:
        return {
            **_slot_record(self.slot),
            "pauli": self.pauli,
        }


def _operation_name(prefix: str, slots: Iterable[SpaceSlot]) -> str:
    fragments = tuple(f"{slot.space.name}_{slot.index}" for slot in slots)
    return ".".join((prefix, *fragments))


def _dependency_names(values) -> tuple[str, ...]:
    values = tuple(values)
    if any(not isinstance(value, (ProductMeasurement, EncodedYInjection))
           for value in values):
        raise TypeError(
            "lattice-surgery dependencies must be typed operation values")
    names = tuple(value.name for value in values)
    if len(set(names)) != len(names):
        raise ValueError("lattice-surgery dependencies must be unique")
    return names


@dataclass(frozen=True, slots=True)
class ProductMeasurement:
    """A nondestructive logical Pauli-product measurement protocol."""

    name: str
    terms: tuple[PauliTerm, ...]
    after: tuple[str, ...] = ()

    def __init__(
        self,
        product: PauliProduct,
        *,
        name: str | None = None,
        after: Iterable["ProductMeasurement | EncodedYInjection"] = (),
    ) -> None:
        if not isinstance(product, PauliProduct):
            raise TypeError(
                "product measurement requires a typed cudaq.logical.PauliProduct; "
                "compose it with cudaq.logical.X/Y/Z(slot)")
        if product.sign != 1:
            raise ValueError(
                "lattice-surgery product measurement does not yet support "
                "a negative PauliProduct")
        if product.identities:
            raise ValueError(
                "lattice-surgery product measurement does not accept "
                "identity-only coverage")
        normalized = tuple(
            PauliTerm(
                _require_slot(
                    factor.operand,
                    what="lattice-surgery Pauli-product operand",
                ),
                factor.pauli,
            ) for factor in product.factors)
        if not normalized:
            raise ValueError("a product measurement requires at least one term")
        slots = tuple(term.slot for term in normalized)
        if len(set(slots)) != len(slots):
            raise ValueError(
                "a product measurement cannot repeat one logical slot")
        derived = _operation_name(
            "mpp_" + "".join(term.pauli.lower() for term in normalized),
            slots,
        )
        resolved_name = derived if name is None else name
        if not isinstance(resolved_name, str) or not resolved_name:
            raise ValueError(
                "product-measurement name must be a nonempty string")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "terms", normalized)
        object.__setattr__(self, "after", _dependency_names(after))

    @property
    def slots(self) -> tuple[SpaceSlot, ...]:
        return tuple(term.slot for term in self.terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "product_measurement",
            "name": self.name,
            "terms": [term.to_dict() for term in self.terms],
            "after": list(self.after),
        }


@dataclass(frozen=True, slots=True, init=False)
class EncodedYInjection:
    """Compiler-derived encoded-|Y> S-injection program.

    Users author this operation as ordinary QLX: nondestructive ``M_ZZ`` on
    data and encoded-|Y>, destructive ``M_X`` of the encoded-|Y>, and a
    conditional logical Z driven by ``M_ZZ xor M_X``.  The lattice-surgery
    problem extractor recognizes that typed SSA structure; this class is the
    transparent provider-facing summary, not an authoring primitive.  It is a
    Clifford analogue of Litinski's measurement-based rotation patterns, not
    the non-Clifford resource protocol shown in that paper's Fig. 7.
    """

    name: str
    target: SpaceSlot
    factory: SpaceSlot
    source_digest: str
    source_artifact: str
    after: tuple[str, ...] = ()

    def __init__(
        self,
        *_args,
        **_kwargs,
    ) -> None:
        raise TypeError(
            "EncodedYInjection is compiler-derived; author an ordinary "
            "@cudaq.logical.program and pass it to "
            "cudaq.logical.qec.lattice_surgery.problem")

    @classmethod
    def _from_verified_source(
        cls,
        *,
        target: SpaceSlot,
        factory: SpaceSlot,
        name: str,
        source_artifact: str,
        after: Iterable[ProductMeasurement | "EncodedYInjection"] = (),
    ) -> "EncodedYInjection":
        target = _require_slot(target, what="encoded-Y injection target")
        factory = _require_slot(factory, what="encoded-Y injection factory")
        if target == factory:
            raise ValueError(
                "encoded-Y injection target and factory must differ")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "encoded-Y injection program name must be a nonempty string")
        if not isinstance(source_artifact, str) or not source_artifact:
            raise TypeError(
                "encoded-Y injection source artifact must be a nonempty "
                "canonical replay bundle")
        try:
            source_bytes = bytes.fromhex(source_artifact)
        except ValueError as exc:
            raise ValueError(
                "encoded-Y injection source artifact must be hexadecimal"
            ) from exc
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        value = object.__new__(cls)
        object.__setattr__(value, "name", name)
        object.__setattr__(value, "target", target)
        object.__setattr__(value, "factory", factory)
        object.__setattr__(value, "source_digest", source_digest)
        object.__setattr__(value, "source_artifact", source_artifact)
        object.__setattr__(value, "after", _dependency_names(after))
        return value

    @property
    def slots(self) -> tuple[SpaceSlot, ...]:
        return self.target, self.factory

    def to_dict(self) -> dict[str, Any]:
        mzz = f"{self.name}.mzz"
        mx = f"{self.name}.mx"
        return {
            "kind":
                "encoded_y_injection_program",
            "name":
                self.name,
            "target":
                _slot_record(self.target),
            "factory":
                _slot_record(self.factory),
            "after":
                list(self.after),
            "source_digest":
                self.source_digest,
            "source_artifact":
                self.source_artifact,
            "instructions": [
                {
                    "kind": "measure_product",
                    "name": mzz,
                    "terms": [
                        {
                            **_slot_record(self.target),
                            "pauli": "Z",
                        },
                        {
                            **_slot_record(self.factory),
                            "pauli": "Z",
                        },
                    ],
                    "destructive": False,
                },
                {
                    "kind": "measure",
                    "name": mx,
                    **_slot_record(self.factory),
                    "basis": "X",
                    "destructive": True,
                },
                {
                    "kind": "conditional_pauli",
                    "name": f"{self.name}.correction",
                    **_slot_record(self.target),
                    "pauli": "Z",
                    "condition": {
                        "kind": "xor",
                        "records": [mzz, mx],
                    },
                },
            ],
        }


class ProgramInstructionKind(str, Enum):
    """Canonical instruction families extracted from one placed P1 program."""

    PREPARE = "prepare"
    PAULI = "pauli"
    MEASURE_PRODUCT = "measure_product"
    MEASURE = "measure"
    XOR = "xor"
    CONDITIONAL_PAULI = "conditional_pauli"
    RETURN = "return"

    __str__ = str.__str__


@dataclass(frozen=True, slots=True)
class ProgramInstruction:
    """One compiler-derived logical instruction in a placed surgery program.

    This is a provider-facing semantic record, not an application authoring
    API.  Its deliberately small closed shape makes P1 program replay
    inspectable without asking an external provider to parse or reinterpret
    QLX MLIR.
    """

    kind: ProgramInstructionKind
    name: str
    slots: tuple[SpaceSlot, ...] = ()
    paulis: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    state: str | None = None
    basis: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProgramInstructionKind):
            raise TypeError("placed-program instruction kind must be "
                            "ProgramInstructionKind")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("placed-program instruction name must be nonempty")
        slots = tuple(
            _require_slot(value, what="placed-program instruction slot")
            for value in self.slots)
        paulis = tuple(str(value).upper() for value in self.paulis)
        if any(value not in {"X", "Y", "Z"} for value in paulis):
            raise ValueError("placed-program Paulis must be X, Y, or Z")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if any(not isinstance(value, str) or not value
               for value in (*inputs, *outputs)):
            raise TypeError("placed-program record names must be nonempty")
        if len(set(outputs)) != len(outputs):
            raise ValueError("placed-program output records must be unique")

        state = self.state
        basis = None if self.basis is None else str(self.basis).upper()
        if self.kind == ProgramInstructionKind.PREPARE:
            valid = (len(slots) == 1 and not paulis and not inputs and
                     not outputs and state in {"zero", "plus"} and
                     basis is None)
        elif self.kind == ProgramInstructionKind.PAULI:
            valid = (len(slots) == len(paulis) == 1 and not inputs and
                     not outputs and state is None and basis is None)
        elif self.kind == ProgramInstructionKind.MEASURE_PRODUCT:
            valid = (bool(slots) and len(slots) == len(paulis) and
                     not inputs and len(outputs) == 1 and state is None and
                     basis is None)
        elif self.kind == ProgramInstructionKind.MEASURE:
            valid = (len(slots) == 1 and not paulis and not inputs and
                     len(outputs) == 1 and state is None and
                     basis in {"X", "Z"})
        elif self.kind == ProgramInstructionKind.XOR:
            valid = (not slots and not paulis and len(inputs) == 2 and
                     len(outputs) == 1 and state is None and basis is None)
        elif self.kind == ProgramInstructionKind.CONDITIONAL_PAULI:
            valid = (len(slots) == len(paulis) == len(inputs) == 1 and
                     not outputs and state is None and basis is None)
        else:
            valid = (self.kind == ProgramInstructionKind.RETURN and
                     not slots and not paulis and bool(inputs) and
                     not outputs and state is None and basis is None)
        if not valid:
            raise ValueError(
                f"invalid fields for placed-program {self.kind.value!r} "
                "instruction")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "paulis", paulis)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "basis", basis)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "name": self.name,
            "slots": [_slot_record(value) for value in self.slots],
            "paulis": list(self.paulis),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "state": self.state,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, value, *, device: Device) -> "ProgramInstruction":
        value = _record(
            value,
            what="placed-program instruction",
            keys=(
                "kind",
                "name",
                "slots",
                "paulis",
                "inputs",
                "outputs",
                "state",
                "basis",
            ),
        )
        try:
            kind = ProgramInstructionKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "unsupported placed-program instruction kind") from exc
        return cls(
            kind=kind,
            name=value["name"],
            slots=tuple(
                _slot_from_record(
                    item,
                    device=device,
                    what="placed-program instruction slot",
                ) for item in _array(
                    value["slots"],
                    what="placed-program instruction slots",
                )),
            paulis=tuple(_array(value["paulis"], what="placed-program Paulis")),
            inputs=tuple(_array(value["inputs"], what="placed-program inputs")),
            outputs=tuple(
                _array(value["outputs"], what="placed-program outputs")),
            state=value["state"],
            basis=value["basis"],
        )


@dataclass(frozen=True, slots=True, init=False)
class LatticeSurgeryProgram:
    """Authenticated provider-facing view of one verified placed P1 program."""

    name: str
    source_digest: str
    source_artifact: str
    instructions: tuple[ProgramInstruction, ...]

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError(
            "LatticeSurgeryProgram is compiler-derived; pass a verified P1 "
            "cudaq.logical.compiler.Build to "
            "cudaq.logical.qec.lattice_surgery.problem")

    @classmethod
    def _from_verified_source(
        cls,
        *,
        name: str,
        source_artifact: str,
        instructions: Iterable[ProgramInstruction],
    ) -> "LatticeSurgeryProgram":
        if not isinstance(name, str) or not name:
            raise ValueError("placed-program name must be nonempty")
        if not isinstance(source_artifact, str) or not source_artifact:
            raise TypeError(
                "placed-program source artifact must be a canonical replay "
                "bundle")
        try:
            source_bytes = bytes.fromhex(source_artifact)
        except ValueError as exc:
            raise ValueError(
                "placed-program source artifact must be hexadecimal") from exc
        instructions = tuple(instructions)
        if (not instructions or any(not isinstance(value, ProgramInstruction)
                                    for value in instructions)):
            raise TypeError(
                "placed programs require typed ProgramInstruction values")
        if instructions[-1].kind != ProgramInstructionKind.RETURN:
            raise ValueError("placed programs must end with one return")
        names = tuple(value.name for value in instructions)
        if len(set(names)) != len(names):
            raise ValueError("placed-program instruction names must be unique")
        defined = set()
        for instruction in instructions:
            missing = set(instruction.inputs) - defined
            if missing:
                raise ValueError(
                    f"placed-program instruction {instruction.name!r} uses "
                    f"undefined record(s) {sorted(missing)!r}")
            overlap = set(instruction.outputs) & defined
            if overlap:
                raise ValueError(
                    "placed-program records must have one definition")
            defined.update(instruction.outputs)
        value = object.__new__(cls)
        object.__setattr__(value, "name", name)
        object.__setattr__(
            value,
            "source_digest",
            hashlib.sha256(source_bytes).hexdigest(),
        )
        object.__setattr__(value, "source_artifact", source_artifact)
        object.__setattr__(value, "instructions", instructions)
        return value

    @property
    def slots(self) -> tuple[SpaceSlot, ...]:
        return tuple(
            dict.fromkeys(slot for instruction in self.instructions
                          for slot in instruction.slots))

    @property
    def returned_records(self) -> tuple[str, ...]:
        return self.instructions[-1].inputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "qlx.lattice_surgery.program/v1",
            "name": self.name,
            "source_digest": self.source_digest,
            "source_artifact": self.source_artifact,
            "instructions": [value.to_dict() for value in self.instructions],
        }


LatticeSurgeryOperation = ProductMeasurement | EncodedYInjection


@dataclass(frozen=True, slots=True)
class LatticeSurgeryProblem:
    """Provider-neutral P1-to-P2 lattice-surgery mapping problem."""

    operations: tuple[LatticeSurgeryOperation, ...]
    program: LatticeSurgeryProgram | None = None
    strategy: str | SchedulingStrategy = scheduling.greedy_asap
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not operations or any(
                not isinstance(value, (ProductMeasurement, EncodedYInjection))
                for value in operations):
            raise TypeError(
                "a lattice-surgery problem requires typed operations")
        if self.program is not None:
            if not isinstance(self.program, LatticeSurgeryProgram):
                raise TypeError(
                    "lattice-surgery problem program must be compiler-derived")
            verified_program, verified_operations = (
                _placed_program_from_artifact(
                    self.program.source_artifact,
                    slots=self.program.slots,
                ))
            if (verified_program.to_dict() != self.program.to_dict() or
                    tuple(value.to_dict() for value in verified_operations)
                    != tuple(value.to_dict() for value in operations)):
                raise ValueError(
                    "placed program does not match its verified canonical P1 "
                    "source artifact")
        names = tuple(value.name for value in operations)
        if len(set(names)) != len(names):
            raise ValueError("lattice-surgery operation names must be unique")
        known = set()
        for value in operations:
            if isinstance(value, EncodedYInjection):
                verified = _encoded_y_injection_from_artifact(
                    value.source_artifact,
                    operands=value.slots,
                    after=(),
                )
                actual = value.to_dict()
                actual["after"] = []
                if verified.to_dict() != actual:
                    raise ValueError(
                        "encoded-Y injection does not match its verified "
                        "canonical P0 source artifact")
            missing = set(value.after) - known
            if missing:
                raise ValueError(
                    f"operation {value.name!r} depends on absent or later "
                    f"operations {sorted(missing)!r}")
            known.add(value.name)
        factory_owners = {}
        for value in operations:
            if not isinstance(value, EncodedYInjection):
                continue
            key = (id(value.factory.space), value.factory.index)
            if key in factory_owners:
                raise ValueError(
                    f"encoded-Y factory slot @{value.factory.space.name}"
                    f"[{value.factory.index}] is consumed by more than one "
                    "encoded-Y injection")
            factory_owners[key] = value
        for value in operations:
            for slot in value.slots:
                owner = factory_owners.get((id(slot.space), slot.index))
                if owner is not None and value is not owner:
                    raise ValueError(
                        f"encoded-Y factory slot @{slot.space.name}"
                        f"[{slot.index}] is exclusively owned by encoded-Y "
                        f"injection {owner.name!r}")
        strategy = self.strategy
        if isinstance(strategy, SchedulingStrategy):
            if not strategy.supports("lattice_surgery"):
                raise ValueError(
                    "lattice-surgery problems require a lattice-surgery "
                    "scheduling strategy")
            strategy = str(strategy)
        if not isinstance(strategy, str) or not strategy:
            raise ValueError(
                "lattice-surgery strategy must be a nonempty string")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "qlx.lattice_surgery.problem/v4",
            "strategy": self.strategy,
            "operations": [value.to_dict() for value in self.operations],
            "program":
                (None if self.program is None else self.program.to_dict()),
            "metadata": _json_value(self.metadata),
        }

    def to_json(self) -> str:
        return _mapping_module().canonical_json({
            **self.to_dict(),
            "digest": self.digest,
        })

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, value, *, device: Device) -> "LatticeSurgeryProblem":
        value = _record(
            value,
            what="lattice-surgery problem",
            keys=("schema", "strategy", "operations", "program", "metadata"),
        )
        if value["schema"] != "qlx.lattice_surgery.problem/v4":
            raise ValueError("unsupported lattice-surgery problem schema")
        strategy = value["strategy"]
        if not isinstance(strategy, str) or not strategy:
            raise TypeError(
                "lattice-surgery problem strategy must be a nonempty string")
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise TypeError(
                "lattice-surgery problem metadata must be a JSON object")
        operations = []
        by_name = {}
        for index, raw in enumerate(
                _array(
                    value["operations"],
                    what="lattice-surgery problem operations",
                )):
            if not isinstance(raw, dict):
                raise TypeError(
                    f"lattice-surgery operation {index} must be a JSON object")
            kind = raw.get("kind")
            if kind == "product_measurement":
                raw = _record(
                    raw,
                    what=f"lattice-surgery operation {index}",
                    keys=("kind", "name", "terms", "after"),
                )
                terms = []
                for term_index, term in enumerate(
                        _array(
                            raw["terms"],
                            what=f"lattice-surgery operation {index} terms",
                        )):
                    term = _record(
                        term,
                        what=(f"lattice-surgery operation {index} term "
                              f"{term_index}"),
                        keys=("space", "index", "pauli"),
                    )
                    slot = _slot_from_record(
                        {
                            "space": term["space"],
                            "index": term["index"],
                        },
                        device=device,
                        what=(f"lattice-surgery operation {index} term "
                              f"{term_index}"),
                    )
                    pauli = term["pauli"]
                    if not isinstance(pauli, str):
                        raise TypeError(
                            "lattice-surgery term pauli must be a string")
                    terms.append((slot, pauli))
                factory = lambda after: ProductMeasurement(
                    PauliProduct(
                        tuple(
                            PauliFactor(slot, pauli) for slot, pauli in terms)),
                    name=raw["name"],
                    after=after,
                )
            elif kind == "encoded_y_injection_program":
                raw = _record(
                    raw,
                    what=f"lattice-surgery operation {index}",
                    keys=(
                        "kind",
                        "name",
                        "target",
                        "factory",
                        "after",
                        "source_digest",
                        "source_artifact",
                        "instructions",
                    ),
                )
                target = _slot_from_record(
                    raw["target"],
                    device=device,
                    what=f"lattice-surgery operation {index} target",
                )
                factory_slot = _slot_from_record(
                    raw["factory"],
                    device=device,
                    what=f"lattice-surgery operation {index} factory",
                )
                factory = lambda after: _encoded_y_injection_from_artifact(
                    raw["source_artifact"],
                    operands=(target, factory_slot),
                    after=after,
                )
            else:
                raise ValueError(
                    f"unsupported lattice-surgery operation kind {kind!r}")
            name = raw["name"]
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "lattice-surgery operation name must be nonempty")
            after_names = _array(
                raw["after"],
                what=f"lattice-surgery operation {index} dependencies",
            )
            if any(not isinstance(item, str) or not item
                   for item in after_names):
                raise TypeError("lattice-surgery dependencies must be strings")
            try:
                after = tuple(by_name[item] for item in after_names)
            except KeyError as exc:
                raise ValueError(
                    f"lattice-surgery operation {name!r} depends on absent "
                    f"or later operation {exc.args[0]!r}") from exc
            operation = factory(after)
            if (kind == "encoded_y_injection_program" and
                    _json_value(operation.to_dict()) != _json_value(raw)):
                raise ValueError(
                    "encoded-Y injection record does not match its verified "
                    "canonical P0 source artifact")
            operations.append(operation)
            by_name[name] = operation
        raw_program = value["program"]
        placed_program = None
        if raw_program is not None:
            raw_program = _record(
                raw_program,
                what="placed lattice-surgery program",
                keys=(
                    "schema",
                    "name",
                    "source_digest",
                    "source_artifact",
                    "instructions",
                ),
            )
            if raw_program["schema"] != "qlx.lattice_surgery.program/v1":
                raise ValueError("unsupported placed-program schema")
            placed_program, extracted = _placed_program_from_artifact(
                raw_program["source_artifact"],
                device=device,
            )
            if (_json_value(
                    placed_program.to_dict()) != _json_value(raw_program) or
                    tuple(value.to_dict() for value in extracted) != tuple(
                        value.to_dict() for value in operations)):
                raise ValueError(
                    "placed-program record does not match its verified "
                    "canonical P1 source artifact")
        return cls(
            operations=tuple(operations),
            program=placed_program,
            strategy=strategy,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, text: str, *, device: Device):
        try:
            payload = _strict_json_loads(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid lattice-surgery problem JSON") from exc
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ValueError("lattice-surgery problem JSON requires a digest")
        claimed = payload.pop("digest")
        if not isinstance(claimed, str) or claimed != _digest(payload):
            raise _mapping_module().MappingVerificationError(
                "lattice-surgery problem digest mismatch")
        return cls.from_dict(payload, device=device)

    @classmethod
    def load(cls, path, *, device: Device):
        return cls.from_json(
            Path(path).read_text(encoding="utf-8"),
            device=device,
        )


class TemporalResourceKind(str, Enum):
    """Provider-neutral resource families used by the temporal mapper."""

    CHANNEL = "channel"
    CHANNEL_PORT = "channel_port"
    BUFFER = "buffer"
    QEC_WORKSPACE = "qec_workspace"
    ENDPOINT_BOUNDARY = "endpoint_boundary"
    EXCLUSIVE = "exclusive"

    __str__ = str.__str__


@dataclass(frozen=True, slots=True)
class TemporalResource:
    """One capacity-bearing P2 resource visible to core temporal mapping."""

    kind: TemporalResourceKind
    name: str
    capacity: int
    provider: str = "qlx"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TemporalResourceKind):
            raise TypeError(
                "temporal resource kind must be TemporalResourceKind")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("temporal resource name must be nonempty")
        if (not isinstance(self.capacity, int) or
                isinstance(self.capacity, bool) or self.capacity <= 0):
            raise TypeError("temporal resource capacity must be a positive int")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("temporal resource provider must be nonempty")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.provider, str(self.kind), self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "name": self.name,
            "capacity": self.capacity,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, value) -> "TemporalResource":
        value = _record(
            value,
            what="temporal resource",
            keys=("kind", "name", "capacity", "provider"),
        )
        try:
            kind = TemporalResourceKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported temporal resource kind") from exc
        return cls(
            kind=kind,
            name=value["name"],
            capacity=value["capacity"],
            provider=value["provider"],
        )


@dataclass(frozen=True, slots=True)
class TemporalResourceClaim:
    """One epoch's immutable use of a normalized temporal resource."""

    resource: TemporalResource
    amount: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.resource, TemporalResource):
            raise TypeError("temporal claims require a TemporalResource")
        if (not isinstance(self.amount, int) or isinstance(self.amount, bool) or
                self.amount <= 0):
            raise TypeError("temporal claim amount must be a positive int")

    def to_dict(self) -> dict[str, Any]:
        return {"resource": self.resource.to_dict(), "amount": self.amount}

    @classmethod
    def from_dict(cls, value) -> "TemporalResourceClaim":
        value = _record(
            value,
            what="temporal resource claim",
            keys=("resource", "amount"),
        )
        return cls(
            resource=TemporalResource.from_dict(value["resource"]),
            amount=value["amount"],
        )


@dataclass(frozen=True, slots=True)
class CompatibilityDiagnostic:
    """One provider-independent explanation for a compatibility decision."""

    code: str
    message: str
    site: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("compatibility diagnostic code must be nonempty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError(
                "compatibility diagnostic message must be nonempty")
        if self.site is not None and (not isinstance(self.site, str) or
                                      not self.site):
            raise ValueError("compatibility diagnostic site must be nonempty")
        object.__setattr__(self, "details", _frozen_mapping(self.details))


@dataclass(frozen=True, slots=True)
class Compatibility:
    """Typed compatibility verdict with preserved provider diagnostics."""

    supported: bool
    diagnostics: tuple[CompatibilityDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise TypeError("compatibility supported flag must be bool")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, CompatibilityDiagnostic)
               for value in diagnostics):
            raise TypeError(
                "compatibility diagnostics must be CompatibilityDiagnostic values"
            )
        if self.supported and diagnostics:
            raise ValueError(
                "a supported compatibility verdict cannot carry diagnostics")
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class QECNetworkValue:
    """Stable typed boundary handle for one P1 SSA generation."""

    name: str
    placement: str
    generation: int
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("network value name must be nonempty")
        if not isinstance(self.placement, str) or not self.placement:
            raise ValueError("network value placement must be nonempty")
        if (not isinstance(self.generation, int) or
                isinstance(self.generation, bool) or self.generation < 0):
            raise TypeError("network value generation must be nonnegative")
        if self.kind not in {"patch", "record"}:
            raise ValueError("network value kind must be 'patch' or 'record'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "placement": self.placement,
            "generation": self.generation,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value) -> "QECNetworkValue":
        value = _record(
            value,
            what="network value",
            keys=("name", "placement", "generation", "kind"),
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class QECNetworkAction:
    """One placed MPP action selected for network compilation."""

    site: Any
    measurement: ProductMeasurement
    owners: tuple[QECBlockOwner, ...]
    blocks: tuple[str, ...]
    after: tuple[str, ...]
    region: str

    def __post_init__(self) -> None:
        from cudaq.logical.qec.lowering import ActionSiteHandle

        if not isinstance(self.site, ActionSiteHandle):
            raise TypeError("network actions require an ActionSiteHandle")
        if not isinstance(self.measurement, ProductMeasurement):
            raise TypeError("network actions require a ProductMeasurement")
        if self.measurement.name != self.site.symbol:
            raise ValueError(
                "network action measurement name must equal its action-site symbol"
            )
        owners = tuple(self.owners)
        if any(not isinstance(value, QECBlockOwner) for value in owners):
            raise TypeError(
                "network action owners must be QECBlockOwner values")
        placements = tuple(value.placement for value in owners)
        if placements != self.site.placements:
            raise ValueError(
                "network action owners must follow the action-site placement order"
            )
        blocks = tuple(self.blocks)
        if (len(blocks) != len(owners) or any(
                not isinstance(value, str) or not value for value in blocks)):
            raise ValueError(
                "network action blocks must identify each encoded owner")
        after = tuple(self.after)
        if after != self.measurement.after:
            raise ValueError(
                "network action dependencies must equal the typed measurement")
        if len(set(after)) != len(after):
            raise ValueError("network action dependencies must be unique")
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("network action region must be nonempty")
        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "after", after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": {
                "symbol": self.site.symbol,
                "kind": self.site.kind,
                "objective_family": self.site.objective_family,
                "objective": self.site.objective,
                "placements": list(self.site.placements),
                "parameters": _json_value(self.site.parameters),
                "input_arity": self.site.input_arity,
                "result_arity": self.site.result_arity,
                "channel": self.site.channel,
                "channel_capability":
                    (None if self.site.channel_capability is None else
                     self.site.channel_capability.key),
                "endpoints": list(self.site.endpoints),
                "direction": self.site.direction,
            },
            "measurement": self.measurement.to_dict(),
            "owners": [{
                "placement": value.placement,
                "logical_index": value.logical_index,
                "source_allocation": value.source_allocation,
                "source_group": value.source_group,
                "source_path": list(value.source_path),
            } for value in self.owners],
            "blocks": list(self.blocks),
            "after": list(self.after),
            "region": self.region,
        }


@dataclass(frozen=True, slots=True)
class QECNetworkRegion:
    """One closed straight-line replacement boundary in a placed program."""

    id: str
    actions: tuple[str, ...]
    live_inputs: tuple[QECNetworkValue, ...]
    live_outputs: tuple[QECNetworkValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("network region id must be nonempty")
        actions = tuple(self.actions)
        if (not actions or any(
                not isinstance(value, str) or not value for value in actions) or
                len(set(actions)) != len(actions)):
            raise ValueError("network regions require unique action names")
        inputs = tuple(self.live_inputs)
        outputs = tuple(self.live_outputs)
        if any(not isinstance(value, QECNetworkValue)
               for value in (*inputs, *outputs)):
            raise TypeError(
                "network region boundaries require QECNetworkValue values")
        if len({value.name for value in inputs}) != len(inputs):
            raise ValueError("network region live inputs must be unique")
        if len({value.name for value in outputs}) != len(outputs):
            raise ValueError("network region live outputs must be unique")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "live_inputs", inputs)
        object.__setattr__(self, "live_outputs", outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actions": list(self.actions),
            "live_inputs": [value.to_dict() for value in self.live_inputs],
            "live_outputs": [value.to_dict() for value in self.live_outputs],
        }

    @classmethod
    def from_dict(cls, value) -> "QECNetworkRegion":
        value = _record(
            value,
            what="network region",
            keys=("id", "actions", "live_inputs", "live_outputs"),
        )
        return cls(
            id=value["id"],
            actions=tuple(value["actions"]),
            live_inputs=tuple(
                QECNetworkValue.from_dict(item)
                for item in value["live_inputs"]),
            live_outputs=tuple(
                QECNetworkValue.from_dict(item)
                for item in value["live_outputs"]),
        )


def _port_record(value: QECChannelPort) -> dict[str, Any]:
    return {
        "name": value.name,
        "region": value.region.name,
        "slot": value.slot,
        "capabilities": [capability.key for capability in value.capabilities],
        "concurrency": value.concurrency,
        "provider": value.provider,
        "metadata": _json_value(value.metadata),
    }


def _channel_record(value: QECChannelRealization) -> dict[str, Any]:
    return {
        "name": value.name,
        "logical_channel": value.logical_channel.name,
        "source": value.source.name,
        "destination": value.destination.name,
        "capabilities": [capability.key for capability in value.capabilities],
        "concurrency": value.concurrency,
        "provider": value.provider,
        "metadata": _json_value(value.metadata),
        "protocol": None if value.protocol is None else value.protocol.name,
    }


def _device_channel_inventory(device: Device):
    """Return the canonical P2 channel inventory and its temporal resources."""

    if not isinstance(device, Device):
        raise TypeError("QEC network channel inventory requires a Device")
    ports = () if device.qec is None else tuple(device.qec.channel_ports)
    channels = () if device.qec is None else tuple(device.qec.channels)
    resources = tuple(
        TemporalResource(
            TemporalResourceKind.CHANNEL,
            channel.name,
            channel.concurrency,
            channel.provider,
        ) for channel in channels) + tuple(
            TemporalResource(
                TemporalResourceKind.CHANNEL_PORT,
                port.name,
                port.concurrency,
                port.provider,
            ) for port in ports)
    return resources, ports, channels


@dataclass(frozen=True, slots=True)
class QECNetworkRequest:
    """Authenticated provider-neutral request for complete network planning."""

    source_sha256: str
    lowering_manifest_sha256: str
    device_architecture_sha256: str
    actions: tuple[QECNetworkAction, ...]
    regions: tuple[QECNetworkRegion, ...]
    resources: tuple[TemporalResource, ...] = ()
    channel_ports: tuple[QECChannelPort, ...] = ()
    channels: tuple[QECChannelRealization, ...] = ()
    policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, what in (
            (self.source_sha256, "network request source digest"),
            (self.lowering_manifest_sha256, "network lowering manifest digest"),
            (self.device_architecture_sha256,
             "network device architecture digest"),
        ):
            _require_commitment_digest(value, what=what)
        actions = tuple(self.actions)
        regions = tuple(self.regions)
        resources = tuple(self.resources)
        ports = tuple(self.channel_ports)
        channels = tuple(self.channels)
        if not actions or any(
                not isinstance(value, QECNetworkAction) for value in actions):
            raise TypeError("network requests require typed actions")
        if not regions or any(
                not isinstance(value, QECNetworkRegion) for value in regions):
            raise TypeError("network requests require typed regions")
        if any(not isinstance(value, TemporalResource) for value in resources):
            raise TypeError(
                "network request resources must be TemporalResource values")
        if any(not isinstance(value, QECChannelPort) for value in ports):
            raise TypeError(
                "network request channel ports must be QECChannelPort values")
        if any(not isinstance(value, QECChannelRealization)
               for value in channels):
            raise TypeError(
                "network request channels must be QECChannelRealization values")
        action_names = tuple(value.site.symbol for value in actions)
        if len(set(action_names)) != len(action_names):
            raise ValueError("network request action sites must be unique")
        region_ids = tuple(value.id for value in regions)
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("network request region ids must be unique")
        covered = tuple(name for region in regions for name in region.actions)
        if sorted(covered) != sorted(action_names) or len(covered) != len(
                action_names):
            raise ValueError(
                "network request regions must cover every action exactly once")
        by_name = {value.site.symbol: value for value in actions}
        for region in regions:
            for name in region.actions:
                if by_name[name].region != region.id:
                    raise ValueError(
                        "network action names the wrong replacement region")
        resource_keys = tuple(value.key for value in resources)
        if len(set(resource_keys)) != len(resource_keys):
            raise ValueError(
                "network request resources must have unique identities")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "channel_ports", ports)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "policy", _frozen_mapping(self.policy))

    @property
    def policy_sha256(self) -> str:
        return _digest(self.policy)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def operations(self) -> tuple[ProductMeasurement, ...]:
        return tuple(value.measurement for value in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "qlx.qec_network.request/v2",
            "source_sha256": self.source_sha256,
            "lowering_manifest_sha256": self.lowering_manifest_sha256,
            "device_architecture_sha256": self.device_architecture_sha256,
            "actions": [value.to_dict() for value in self.actions],
            "regions": [value.to_dict() for value in self.regions],
            "resources": [value.to_dict() for value in self.resources],
            "channel_ports": [
                _port_record(value) for value in self.channel_ports
            ],
            "channels": [_channel_record(value) for value in self.channels],
            "policy": _json_value(self.policy),
        }

    def to_json(self) -> str:
        return _mapping_module().canonical_json({
            **self.to_dict(), "digest": self.digest
        })

    @classmethod
    def from_dict(cls, value, *, device: Device) -> "QECNetworkRequest":
        from cudaq.logical.qec.lowering import ActionSiteHandle
        from cudaq.logical.algebra.pauli import (
            PauliFactor,
            PauliProduct,
        )
        from cudaq.logical.architecture.logical import capability

        value = _record(
            value,
            what="QEC network request",
            keys=(
                "schema",
                "source_sha256",
                "lowering_manifest_sha256",
                "device_architecture_sha256",
                "actions",
                "regions",
                "resources",
                "channel_ports",
                "channels",
                "policy",
            ),
        )
        if value["schema"] != "qlx.qec_network.request/v2":
            raise ValueError("unsupported QEC network request schema")
        if not isinstance(device, Device):
            raise TypeError("QEC network request replay requires a Device")
        known_measurements = {}
        actions = []
        for raw in value["actions"]:
            raw = _record(
                raw,
                what="QEC network action",
                keys=(
                    "site",
                    "measurement",
                    "owners",
                    "blocks",
                    "after",
                    "region",
                ),
            )
            site = raw["site"]
            site = _record(
                site,
                what="QEC network action site",
                keys=(
                    "symbol",
                    "kind",
                    "objective_family",
                    "objective",
                    "placements",
                    "parameters",
                    "input_arity",
                    "result_arity",
                    "channel",
                    "channel_capability",
                    "endpoints",
                    "direction",
                ),
            )
            measurement = raw["measurement"]
            measurement = _record(
                measurement,
                what="QEC network product measurement",
                keys=("kind", "name", "terms", "after"),
            )
            if measurement["kind"] != "product_measurement":
                raise ValueError(
                    "QEC network actions require product measurements")
            try:
                dependencies = tuple(
                    known_measurements[name] for name in measurement["after"])
            except KeyError as exc:
                raise ValueError(
                    "QEC network action depends on an absent or later action"
                ) from exc
            product = PauliProduct(
                tuple(
                    PauliFactor(
                        _slot_from_record(
                            {
                                "space": term["space"],
                                "index": term["index"],
                            },
                            device=device,
                            what="QEC network Pauli term",
                        ),
                        term["pauli"],
                    ) for term in measurement["terms"]))
            typed_measurement = ProductMeasurement(
                product,
                name=measurement["name"],
                after=dependencies,
            )
            handle = ActionSiteHandle(
                symbol=site["symbol"],
                kind=site["kind"],
                objective_family=site["objective_family"],
                objective=site["objective"],
                placements=tuple(site["placements"]),
                parameters=site["parameters"],
                input_arity=site["input_arity"],
                result_arity=site["result_arity"],
                channel=site["channel"],
                channel_capability=(None
                                    if site["channel_capability"] is None else
                                    capability(site["channel_capability"])),
                endpoints=tuple(site["endpoints"]),
                direction=site["direction"],
            )
            owners = tuple(
                QECBlockOwner(
                    placement=owner["placement"],
                    logical_index=owner["logical_index"],
                    source_allocation=owner["source_allocation"],
                    source_group=owner["source_group"],
                    source_path=tuple(owner["source_path"]),
                ) for owner in raw["owners"])
            action = QECNetworkAction(
                site=handle,
                measurement=typed_measurement,
                owners=owners,
                blocks=tuple(raw["blocks"]),
                after=tuple(raw["after"]),
                region=raw["region"],
            )
            actions.append(action)
            known_measurements[typed_measurement.name] = typed_measurement

        resources = tuple(
            TemporalResource.from_dict(item) for item in value["resources"])
        expected_resources, ports, channels = _device_channel_inventory(device)
        canonical_json = _mapping_module().canonical_json
        if canonical_json([_port_record(item) for item in ports
                          ]) != canonical_json(value["channel_ports"]):
            raise ValueError(
                "QEC network channel-port inventory differs from the device")
        if canonical_json([_channel_record(item) for item in channels
                          ]) != canonical_json(value["channels"]):
            raise ValueError(
                "QEC network channel inventory differs from the device")
        if resources != expected_resources:
            raise ValueError(
                "QEC network temporal channel inventory differs from the device"
            )
        return cls(
            source_sha256=value["source_sha256"],
            lowering_manifest_sha256=value["lowering_manifest_sha256"],
            device_architecture_sha256=value["device_architecture_sha256"],
            actions=tuple(actions),
            regions=tuple(
                QECNetworkRegion.from_dict(item) for item in value["regions"]),
            resources=resources,
            channel_ports=ports,
            channels=channels,
            policy=value["policy"],
        )

    @classmethod
    def from_json(cls, text: str, *, device: Device) -> "QECNetworkRequest":
        payload = _network_request_payload(text)
        request = cls.from_dict(payload, device=device)
        if request.digest != _digest(payload):
            raise ValueError(
                "QEC network request reconstructs to noncanonical content")
        return request


def _network_request_payload(text: str) -> dict[str, Any]:
    """Verify and return the provider-neutral JSON envelope without a device."""

    payload = _strict_json_loads(text)
    if not isinstance(payload, dict) or "digest" not in payload:
        raise ValueError("QEC network request JSON requires a digest")
    claimed = payload.pop("digest")
    if claimed != _digest(payload):
        raise ValueError("QEC network request digest mismatch")
    payload = _record(
        payload,
        what="QEC network request",
        keys=(
            "schema",
            "source_sha256",
            "lowering_manifest_sha256",
            "device_architecture_sha256",
            "actions",
            "regions",
            "resources",
            "channel_ports",
            "channels",
            "policy",
        ),
    )
    if payload.get("schema") != "qlx.qec_network.request/v2":
        raise ValueError("unsupported QEC network request schema")
    return payload


@dataclass(frozen=True, slots=True)
class QECNetworkArtifact:
    """One authenticated provider-specific geometry artifact."""

    schema: str
    media_type: str
    payload: Mapping[str, Any]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or not self.schema:
            raise ValueError("provider artifact schema must be nonempty")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("provider artifact media type must be nonempty")
        payload = _frozen_mapping(self.payload)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "sha256", _digest(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "media_type": self.media_type,
            "payload": _json_value(self.payload),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value) -> "QECNetworkArtifact":
        value = _record(
            value,
            what="provider artifact",
            keys=("schema", "media_type", "payload", "sha256"),
        )
        result = cls(
            schema=value["schema"],
            media_type=value["media_type"],
            payload=value["payload"],
        )
        if result.sha256 != value["sha256"]:
            raise ValueError("provider artifact digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class QECNetworkEpoch:
    """Canonical provider-complete encoded-space temporal epoch."""

    id: str
    region: str
    actions: tuple[str, ...]
    claims: tuple[TemporalResourceClaim, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("network epoch id must be nonempty")
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("network epoch region must be nonempty")
        actions = tuple(self.actions)
        if (not actions or any(
                not isinstance(value, str) or not value for value in actions) or
                len(set(actions)) != len(actions)):
            raise ValueError("network epochs require unique action names")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "claims",
                           _normalize_temporal_claims(self.claims))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "region": self.region,
            "actions": list(self.actions),
            "claims": [value.to_dict() for value in self.claims],
        }

    @classmethod
    def from_dict(cls, value) -> "QECNetworkEpoch":
        value = _record(
            value,
            what="QEC network epoch",
            keys=("id", "region", "actions", "claims"),
        )
        return cls(
            id=value["id"],
            region=value["region"],
            actions=tuple(value["actions"]),
            claims=tuple(
                TemporalResourceClaim.from_dict(item)
                for item in value["claims"]),
        )


@dataclass(frozen=True, slots=True)
class QECNetworkPlan:
    """Canonical envelope around one complete provider temporal plan."""

    request_sha256: str
    lowering_manifest_sha256: str
    device_architecture_sha256: str
    policy_sha256: str
    provider_key: str
    required_projector_key: str
    required_projector_pipeline_sha256: str
    epochs: tuple[QECNetworkEpoch, ...]
    artifact: QECNetworkArtifact

    def __post_init__(self) -> None:
        for value, what in (
            (self.request_sha256, "network plan request digest"),
            (self.lowering_manifest_sha256, "network plan manifest digest"),
            (self.device_architecture_sha256,
             "network plan architecture digest"),
            (self.policy_sha256, "network plan policy digest"),
            (
                self.required_projector_pipeline_sha256,
                "network plan projector-pipeline digest",
            ),
        ):
            _require_commitment_digest(value, what=what)
        for value, what in (
            (self.provider_key, "network plan provider key"),
            (self.required_projector_key, "network plan projector key"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{what} must be nonempty")
        epochs = tuple(self.epochs)
        if not epochs or any(
                not isinstance(value, QECNetworkEpoch) for value in epochs):
            raise TypeError("network plans require typed epochs")
        if len({value.id for value in epochs}) != len(epochs):
            raise ValueError("network plan epoch ids must be unique")
        if not isinstance(self.artifact, QECNetworkArtifact):
            raise TypeError("network plans require a QECNetworkArtifact")
        object.__setattr__(self, "epochs", epochs)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "qlx.qec_network.plan/v2",
            "request_sha256": self.request_sha256,
            "lowering_manifest_sha256": self.lowering_manifest_sha256,
            "device_architecture_sha256": self.device_architecture_sha256,
            "policy_sha256": self.policy_sha256,
            "provider_key": self.provider_key,
            "required_projector_key": self.required_projector_key,
            "required_projector_pipeline_sha256":
                (self.required_projector_pipeline_sha256),
            "epochs": [value.to_dict() for value in self.epochs],
            "artifact": self.artifact.to_dict(),
        }

    def to_json(self) -> str:
        return _mapping_module().canonical_json({
            **self.to_dict(), "digest": self.digest
        })

    @classmethod
    def from_dict(cls, value) -> "QECNetworkPlan":
        value = _record(
            value,
            what="QEC network plan",
            keys=(
                "schema",
                "request_sha256",
                "lowering_manifest_sha256",
                "device_architecture_sha256",
                "policy_sha256",
                "provider_key",
                "required_projector_key",
                "required_projector_pipeline_sha256",
                "epochs",
                "artifact",
            ),
        )
        if value["schema"] != "qlx.qec_network.plan/v2":
            raise ValueError("unsupported QEC network plan schema")
        return cls(
            request_sha256=value["request_sha256"],
            lowering_manifest_sha256=value["lowering_manifest_sha256"],
            device_architecture_sha256=value["device_architecture_sha256"],
            policy_sha256=value["policy_sha256"],
            provider_key=value["provider_key"],
            required_projector_key=value["required_projector_key"],
            required_projector_pipeline_sha256=(
                value["required_projector_pipeline_sha256"]),
            epochs=tuple(
                QECNetworkEpoch.from_dict(item) for item in value["epochs"]),
            artifact=QECNetworkArtifact.from_dict(value["artifact"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "QECNetworkPlan":
        payload = _strict_json_loads(text)
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ValueError("QEC network plan JSON requires a digest")
        claimed = payload.pop("digest")
        if claimed != _digest(payload):
            raise ValueError("QEC network plan digest mismatch")
        return cls.from_dict(payload)


def validate_network_plan(
    request: QECNetworkRequest,
    plan: QECNetworkPlan,
) -> None:
    """Validate provider-independent coverage, order, ownership, and claims."""

    if not isinstance(request, QECNetworkRequest):
        raise TypeError("network-plan validation requires a QECNetworkRequest")
    if not isinstance(plan, QECNetworkPlan):
        raise TypeError("network-plan validation requires a QECNetworkPlan")
    commitments = (
        (plan.request_sha256, request.digest, "request"),
        (
            plan.lowering_manifest_sha256,
            request.lowering_manifest_sha256,
            "lowering manifest",
        ),
        (
            plan.device_architecture_sha256,
            request.device_architecture_sha256,
            "device architecture",
        ),
        (plan.policy_sha256, request.policy_sha256, "policy"),
    )
    for actual, expected, what in commitments:
        if actual != expected:
            raise ValueError(f"network plan {what} commitment differs")
    expected = tuple(value.site.symbol for value in request.actions)
    scheduled = tuple(name for epoch in plan.epochs for name in epoch.actions)
    if sorted(scheduled) != sorted(expected) or len(scheduled) != len(expected):
        raise ValueError("network plan must schedule every action exactly once")
    region_by_action = {
        name: region.id for region in request.regions for name in region.actions
    }
    epoch_of = {
        name: index for index, epoch in enumerate(plan.epochs)
        for name in epoch.actions
    }
    actions = {value.site.symbol: value for value in request.actions}
    resources = {value.key: value for value in request.resources}
    for epoch in plan.epochs:
        if any(region_by_action[name] != epoch.region
               for name in epoch.actions):
            raise ValueError(
                "network epoch crosses a replacement-region boundary")
        occupied = set()
        for name in epoch.actions:
            action = actions[name]
            blocks = set(action.blocks)
            if len(blocks) != len(action.blocks) or blocks & occupied:
                raise ValueError("network epoch reuses one encoded block")
            occupied.update(blocks)
            for dependency in action.after:
                if epoch_of[dependency] >= epoch_of[name]:
                    raise ValueError(
                        "network plan reverses or co-schedules an action dependency"
                    )
        claimed = {}
        for claim in epoch.claims:
            declared = resources.get(claim.resource.key)
            if declared is None or declared != claim.resource:
                raise ValueError(
                    "network plan claim references an undeclared temporal resource"
                )
            claimed[claim.resource.key] = (claimed.get(claim.resource.key, 0) +
                                           claim.amount)
            if claimed[claim.resource.key] > declared.capacity:
                raise ValueError(
                    "network plan temporal resource claim exceeds capacity")


def _validate_network_request_selection(
    request: QECNetworkRequest,
    selection,
) -> None:
    """Authenticate request action/block ownership against a QEC witness."""

    if selection is None:
        raise ValueError("canonical network request requires a QEC selection")
    if request.lowering_manifest_sha256 != selection.network_manifest_sha256:
        raise ValueError(
            "P2 network request manifest differs from its QEC selection")
    selected_actions = {
        value.site: value
        for value in selection.actions
        if value.manifest_sha256 == selection.network_manifest_sha256
    }
    owners_by_placement = {}
    for block in selection.blocks:
        for owner in block.owners:
            owners_by_placement[owner.placement] = (block.block, owner)
    if set(selected_actions) != {
            value.site.symbol for value in request.actions
    }:
        raise ValueError(
            "P2 network request action inventory differs from its QEC selection"
        )
    for action in request.actions:
        selected = selected_actions[action.site.symbol]
        if (action.site.kind != selected.kind or
                action.site.objective != selected.objective or
                action.site.placements != selected.placements or
                action.site.channel != selected.channel or
            (None if action.site.channel_capability is None else
             action.site.channel_capability.key) != selected.channel_capability
                or action.site.endpoints != selected.endpoints or
                action.site.direction != selected.direction):
            raise ValueError(
                f"planned action {action.site.symbol!r} differs from its "
                "QEC selection witness")
        try:
            expected_owners = tuple(owners_by_placement[placement][1]
                                    for placement in selected.placements)
            expected_blocks = tuple(owners_by_placement[placement][0]
                                    for placement in selected.placements)
        except KeyError as exc:
            raise ValueError(
                f"planned action {action.site.symbol!r} references an absent "
                "QEC block owner") from exc
        if action.owners != expected_owners or action.blocks != expected_blocks:
            raise ValueError(
                f"planned action {action.site.symbol!r} has different encoded "
                "block ownership")


def _validate_network_request_regions(
    request: QECNetworkRequest,
    runs: Iterable[Iterable[str]],
) -> None:
    """Authenticate maximal replacement boundaries against retained P1."""

    actions = {value.site.symbol: value for value in request.actions}
    expected = []
    for index, raw_run in enumerate(runs):
        run = tuple(raw_run)
        region_id = f"region{index}"
        placements = tuple(
            dict.fromkeys(placement for name in run
                          for placement in actions[name].site.placements))
        expected.append(
            QECNetworkRegion(
                id=region_id,
                actions=run,
                live_inputs=tuple(
                    QECNetworkValue(
                        name=f"{region_id}.in.{placement}",
                        placement=placement,
                        generation=index,
                        kind="patch",
                    ) for placement in placements),
                live_outputs=tuple(
                    QECNetworkValue(
                        name=f"{region_id}.out.{placement}",
                        placement=placement,
                        generation=index + 1,
                        kind="patch",
                    ) for placement in placements),
            ))
    if request.regions != tuple(expected):
        raise ValueError(
            "P2 network request replacement regions differ from retained P1")


@dataclass(frozen=True, slots=True)
class QECNetworkProjection:
    """Authenticated core-derived input to one exact P2-to-P3 projector."""

    source: Any
    device: Device
    request: QECNetworkRequest
    plan: QECNetworkPlan
    problem: LatticeSurgeryProblem
    operation_for_site: Mapping[str, ProductMeasurement]

    def __post_init__(self) -> None:
        from ...compiler import Build

        if not isinstance(self.source, Build) or self.source.stage != P2:
            raise TypeError(
                "network projection requires a P2 cudaq.logical.Build")
        if not isinstance(self.device, Device):
            raise TypeError(
                "network projection requires a cudaq.logical.Device")
        validate_network_plan(self.request, self.plan)
        if not isinstance(self.problem, LatticeSurgeryProblem):
            raise TypeError(
                "network projection requires a typed logical problem")
        mapping = dict(self.operation_for_site)
        if set(mapping) != {
                value.site.symbol for value in self.request.actions
        }:
            raise ValueError(
                "network projection site map must cover every planned action")
        object.__setattr__(self, "operation_for_site",
                           MappingProxyType(mapping))


def _network_compiler_for_manifest(
    device: Device,
    manifest_sha256: str,
):
    """Resolve one device compiler and lowering from an exact manifest."""

    if not isinstance(device, Device):
        raise TypeError(
            "network-plan replay requires a typed cudaq.logical.Device")
    candidates = []
    for compiler in device.compilers:
        if not isinstance(compiler, QECNetworkCompiler):
            continue
        matches = tuple(lowering for lowering in tuple(compiler.qec_lowerings)
                        if isinstance(lowering, QECLowering) and
                        lowering.manifest_sha256 == manifest_sha256)
        if len(matches) > 1:
            raise LookupError(
                f"network compiler {compiler.key!r} contributes the same "
                "QEC manifest more than once")
        if matches:
            candidates.append((compiler, matches[0]))
    if not candidates:
        raise LookupError(
            "device has no QEC network compiler for selected manifest "
            f"{manifest_sha256!r}")
    if len(candidates) != 1:
        raise LookupError(
            "selected QEC network manifest resolves to several compilers: " +
            ", ".join(compiler.key for compiler, _lowering in candidates))
    return candidates[0]


def _retained_network_p1(source, *, device: Device):
    """Reconstruct the authenticated P1 view retained by a canonical P2."""

    from ...compiler import Build, CompilationContext

    transaction = CompilationContext.replay(source)
    return Build(
        context=transaction.context,
        module=transaction.module,
        root=DefinitionHandle(
            source.qec_selection.input_p1,
            "kernel",
            "p1",
        ),
        profile="p1",
        pipeline=None,
        evidence=(),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        placement=source.placement,
        qec_selection=None,
        device=device,
        source_modules=source.source_modules,
    )


def _authenticate_network_replay(source, *, device: Device):
    """Authenticate a canonical P2 request, plan, and planning provider.

    The selected QEC manifest resolves the planner.  The serialized provider
    key is evidence to check, never a dispatch selector.  Core authenticates
    the retained provider-valid plan before an independent physical projector
    is chosen.
    """

    from ...compiler import Build
    from ...compiler.build import _qec_selection_sha256
    from ...compiler.qec_lower import _network_qec_selection

    if not isinstance(source, Build) or source.stage != P2:
        raise TypeError("network-plan replay requires a P2 cudaq.logical.Build")
    if source.qec_selection is None:
        raise ValueError("canonical network P2 requires a QEC selection")
    # Synchronize the immutable snapshot before consulting a derived cache.
    source.module
    cache_key = ("authenticated_qec_network_replay", id(device))
    cached = source._cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        root = source.definitions[source.root.symbol].op
        request = QECNetworkRequest.from_json(
            _attribute_text(root.attributes["qlx.qec_network_request"]),
            device=device,
        )
        plan = QECNetworkPlan.from_json(
            _attribute_text(root.attributes["qlx.qec_network_plan"]))
        metadata = root.attributes["metadata"]
    except KeyError as exc:
        raise ValueError(
            "canonical network P2 requires its request and plan") from exc

    validate_network_plan(request, plan)
    compiler, lowering = _network_compiler_for_manifest(
        device,
        request.lowering_manifest_sha256,
    )
    if plan.provider_key != compiler.key:
        raise ValueError(
            "network plan provider differs from the compiler selected by its "
            "QEC manifest")
    try:
        retained_provider = _attribute_text(metadata["network_provider"])
    except KeyError as exc:
        raise ValueError(
            "canonical network P2 omitted its planning-provider identity"
        ) from exc
    if retained_provider != compiler.key:
        raise ValueError(
            "P2 network provider metadata differs from the selected compiler")
    architecture_digest = _require_digest(
        compiler.architecture_digest(device),
        what=f"network compiler {compiler.key!r} architecture digest",
    )
    if architecture_digest != request.device_architecture_sha256:
        raise ValueError(
            "network request architecture differs from the selected compiler")

    p1 = _retained_network_p1(source, device=device)
    selected = _network_qec_selection(
        p1,
        device=device,
        policy=request.policy,
        expected_compiler=compiler,
    )
    if selected is None or selected.lowering is not lowering:
        raise ValueError(
            "retained P1 does not select the exact QEC network manifest")
    context = QECNetworkContext(
        source=p1,
        lowering=lowering,
        selection=selected.witness,
        selection_digest=_qec_selection_sha256(selected.witness),
        device=device,
        policy=request.policy,
    )
    compatibility = compiler.check_compatibility(request, context)
    if not isinstance(compatibility, Compatibility):
        raise TypeError(
            f"network compiler {compiler.key!r} check_compatibility() must "
            "return Compatibility")
    if not compatibility.supported:
        details = "; ".join(f"{value.code}: {value.message}"
                            for value in compatibility.diagnostics)
        raise ValueError(
            f"network compiler {compiler.key!r} no longer accepts the "
            "retained request" + ("" if not details else f": {details}"))
    result = (request, plan, compiler, context)
    source._cache[cache_key] = result
    return result


def network_projection(source, *, device: Device) -> QECNetworkProjection:
    """Replay the exact canonical network plan retained by one P2 Build.

    Core re-derives the full logical instruction stream from the retained,
    verifier-closed P0/P1 lineage.  The provider receives typed values and does
    not inspect textual MLIR or rely on a second stored lattice-surgery plan.
    """

    from ...compiler import Build

    if not isinstance(source, Build) or source.stage != P2:
        raise TypeError("network projection requires a P2 cudaq.logical.Build")
    if not isinstance(device, Device):
        raise TypeError("network projection requires a cudaq.logical.Device")
    request, plan, _compiler, _context = _authenticate_network_replay(
        source,
        device=device,
    )
    root = source.definitions[source.root.symbol].op
    metadata = root.attributes["metadata"]
    if (_attribute_text(metadata["network_request_sha256"]) != request.digest or
            _attribute_text(metadata["network_plan_sha256"]) != plan.digest or
            _attribute_text(metadata["required_projector"])
            != plan.required_projector_key or
            _attribute_text(metadata["required_projector_pipeline_sha256"])
            != plan.required_projector_pipeline_sha256):
        raise ValueError("P2 network projection metadata differs from its plan")

    from ...compiler.build import (
        _qec_network_region_runs,
        _qec_network_source_sha256,
    )

    expected_source = _qec_network_source_sha256(
        source.module,
        source.qec_selection.input_p1,
        source.placement,
        source.qec_selection,
    )
    if request.source_sha256 != expected_source:
        raise ValueError(
            "P2 network request source commitment differs from retained P1")

    _validate_network_request_selection(request, source.qec_selection)
    _validate_network_request_regions(
        request,
        _qec_network_region_runs(
            source.module,
            source.qec_selection.input_p1,
            (value.site.symbol for value in request.actions),
        ),
    )

    program, operations = _placed_program_from_build(source, device=device)
    if len(operations) != len(request.actions):
        raise ValueError(
            "retained logical program differs from the planned MPP inventory")
    operation_for_site = {
        action.site.symbol: operation
        for action, operation in zip(request.actions, operations)
    }
    site_for_operation = {
        operation.name: action.site.symbol
        for action, operation in zip(request.actions, operations)
    }
    for action, operation in zip(request.actions, operations):
        if tuple(term.to_dict() for term in action.measurement.terms) != tuple(
                term.to_dict() for term in operation.terms):
            raise ValueError(
                f"planned action {action.site.symbol!r} differs from retained P1"
            )
        expected_after = tuple(
            site_for_operation[dependency] for dependency in operation.after)
        if action.after != expected_after:
            raise ValueError(
                f"planned action {action.site.symbol!r} dependencies differ "
                "from retained P1")
    problem = LatticeSurgeryProblem(
        operations=operations,
        program=program,
        strategy=scheduling.greedy_asap,
        metadata={"network_request_sha256": request.digest},
    )
    return QECNetworkProjection(
        source=source,
        device=device,
        request=request,
        plan=plan,
        problem=problem,
        operation_for_site=operation_for_site,
    )


@dataclass(frozen=True, slots=True)
class QECNetworkRegionPlan:
    """Read-only provider view of one validated replacement region."""

    request: QECNetworkRequest
    plan: QECNetworkPlan
    region: QECNetworkRegion

    def __post_init__(self) -> None:
        if not isinstance(self.request, QECNetworkRequest):
            raise TypeError("network region plans require a request")
        if not isinstance(self.plan, QECNetworkPlan):
            raise TypeError("network region plans require a plan")
        if not isinstance(self.region, QECNetworkRegion):
            raise TypeError("network region plans require a region")
        if self.region not in self.request.regions:
            raise ValueError("network region plan references a foreign region")
        validate_network_plan(self.request, self.plan)

    @property
    def actions(self) -> tuple[QECNetworkAction, ...]:
        by_name = {value.site.symbol: value for value in self.request.actions}
        return tuple(by_name[name] for name in self.region.actions)

    @property
    def epochs(self) -> tuple[QECNetworkEpoch, ...]:
        return tuple(value for value in self.plan.epochs
                     if value.region == self.region.id)


class QECNetworkRegionBuilder:
    """Prepared public P2 emission context owned by core QLX.

    Providers contribute an ordinary typed :class:`ProtocolDefinition`.  Core
    owns materialization, live SSA values, result binding, provenance, facets,
    and the final :class:`Build`.
    """

    __slots__ = (
        "_region_plan",
        "_device",
        "_encodings",
        "_emissions",
    )

    def __init__(self, region_plan, *, device, encodings) -> None:
        if not isinstance(region_plan, QECNetworkRegionPlan):
            raise TypeError(
                "prepared region builder requires QECNetworkRegionPlan")
        if not isinstance(device, Device):
            raise TypeError("prepared region builder requires a Device")
        encodings = dict(encodings)
        if any(not isinstance(name, str) or not isinstance(value, Encoding)
               for name, value in encodings.items()):
            raise TypeError(
                "prepared region encodings require placement/Encoding pairs")
        self._region_plan = region_plan
        self._device = device
        self._encodings = MappingProxyType(encodings)
        self._emissions = []

    @property
    def region_plan(self) -> QECNetworkRegionPlan:
        return self._region_plan

    @property
    def device(self) -> Device:
        return self._device

    @property
    def encodings(self) -> Mapping[str, Encoding]:
        return self._encodings

    def realize_epoch(
        self,
        epoch: QECNetworkEpoch,
        protocol: ProtocolDefinition,
        *,
        placements: Iterable[str],
        actions: Iterable[str],
    ) -> None:
        """Select one typed protocol boundary in canonical epoch order."""

        expected_epochs = self._region_plan.epochs
        index = len(self._emissions)
        if index >= len(expected_epochs):
            raise RuntimeError("a network region emitted too many epochs")
        if epoch is not expected_epochs[index] and epoch != expected_epochs[
                index]:
            raise ValueError("network epochs must be emitted in plan order")
        if not isinstance(protocol, ProtocolDefinition):
            raise TypeError(
                "network epoch realization requires a ProtocolDefinition")
        placements = tuple(placements)
        actions = tuple(actions)
        expected_placements = tuple(
            dict.fromkeys(
                placement for action in self._region_plan.request.actions
                if action.site.symbol in epoch.actions
                for placement in action.site.placements))
        if placements != expected_placements:
            raise ValueError(
                "network epoch protocol placements differ from its live owners")
        if actions != epoch.actions:
            raise ValueError(
                "network epoch protocol action results differ from its plan")
        self._emissions.append((epoch, protocol, placements, actions))

    def realize(
        self,
        protocol: ProtocolDefinition,
        *,
        placements: Iterable[str],
        actions: Iterable[str],
    ) -> None:
        """Compatibility spelling for a one-epoch network region."""

        epochs = self._region_plan.epochs
        if len(epochs) != 1:
            raise ValueError(
                "multi-epoch providers must call realize_epoch() explicitly")
        self.realize_epoch(epochs[0],
                           protocol,
                           placements=placements,
                           actions=actions)

    def _take_emissions(self):
        if len(self._emissions) != len(self._region_plan.epochs):
            raise ValueError(
                "network compiler did not realize every selected epoch")
        return tuple(self._emissions)


def _normalize_temporal_claims(
    values: Iterable[TemporalResourceClaim],
) -> tuple[TemporalResourceClaim, ...]:
    claims = tuple(values)
    if any(not isinstance(value, TemporalResourceClaim) for value in claims):
        raise TypeError("epoch claims must be TemporalResourceClaim values")
    totals: dict[tuple[str, str, str], int] = {}
    resources: dict[tuple[str, str, str], TemporalResource] = {}
    for claim in claims:
        key = claim.resource.key
        previous = resources.setdefault(key, claim.resource)
        if previous != claim.resource:
            raise ValueError(
                f"temporal resource {key!r} has conflicting declarations")
        totals[key] = totals.get(key, 0) + claim.amount
    overflow = tuple((key, amount, resources[key].capacity)
                     for key, amount in totals.items()
                     if amount > resources[key].capacity)
    if overflow:
        key, amount, capacity = overflow[0]
        raise PlacementInfeasible(
            f"temporal resource {key!r} claims {amount}, capacity {capacity}")
    return tuple(
        TemporalResourceClaim(resources[key], totals[key])
        for key in sorted(totals))


@dataclass(frozen=True, slots=True)
class LatticeSurgeryEpoch:
    """Provider-independent temporal placement summary."""

    index: int
    operations: tuple[str, ...]
    provider_digest: str
    claims: tuple[TemporalResourceClaim, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.index, int) or isinstance(self.index, bool) or
                self.index < 0):
            raise TypeError(
                "lattice-surgery epoch index must be a nonnegative int")
        operations = tuple(self.operations)
        if (not operations or any(
                not isinstance(value, str) or not value for value in operations)
                or len(set(operations)) != len(operations)):
            raise ValueError(
                "lattice-surgery epochs require unique operation names")
        _require_digest(
            self.provider_digest,
            what="lattice-surgery epoch provider digest",
        )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "claims",
                           _normalize_temporal_claims(self.claims))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operations": list(self.operations),
            "provider_digest": self.provider_digest,
            "claims": [value.to_dict() for value in self.claims],
        }


@dataclass(frozen=True, slots=True)
class EpochPlanningContext:
    """Provider-neutral state visible while constructing one temporal epoch."""

    problem: LatticeSurgeryProblem
    index: int
    completed: frozenset[str]
    remaining: tuple[LatticeSurgeryOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.problem, LatticeSurgeryProblem):
            raise TypeError(
                "epoch planning context requires a lattice-surgery problem")
        if (not isinstance(self.index, int) or isinstance(self.index, bool) or
                self.index < 0):
            raise TypeError("epoch planning index must be a nonnegative int")
        completed = frozenset(self.completed)
        if any(not isinstance(value, str) or not value for value in completed):
            raise TypeError("completed lattice-surgery operations need names")
        remaining = tuple(self.remaining)
        if any(value not in self.problem.operations for value in remaining):
            raise ValueError(
                "epoch planning context contains an operation outside its "
                "problem")
        remaining_names = {value.name for value in remaining}
        if completed & remaining_names:
            raise ValueError(
                "completed and remaining lattice-surgery operations overlap")
        if completed | remaining_names != {
                value.name for value in self.problem.operations
        }:
            raise ValueError(
                "epoch planning context must partition its problem")
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "remaining", remaining)


@dataclass(frozen=True, slots=True)
class EpochCandidate:
    """One provider-proven realization of an entire simultaneous epoch."""

    operations: tuple[str, ...]
    witness: Mapping[str, Any]
    claims: tuple[TemporalResourceClaim, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    next_state: Any = field(default=None, repr=False, compare=False)
    _provider_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if (not operations or any(
                not isinstance(value, str) or not value for value in operations)
                or len(set(operations)) != len(operations)):
            raise ValueError(
                "epoch candidates require unique lattice-surgery operation "
                "names")
        object.__setattr__(self, "operations", operations)
        witness = _frozen_mapping(self.witness)
        claims = _normalize_temporal_claims(self.claims)
        object.__setattr__(self, "witness", witness)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))
        object.__setattr__(
            self,
            "_provider_digest",
            _digest({
                "witness": witness,
                "claims": [value.to_dict() for value in claims],
            }),
        )

    @property
    def provider_digest(self) -> str:
        return self._provider_digest


@dataclass(frozen=True, slots=True)
class EpochInfeasible:
    """Typed rejection of one complete candidate batch by a provider."""

    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("epoch infeasibility requires a nonempty reason")
        object.__setattr__(self, "evidence", _frozen_mapping(self.evidence))


@dataclass(frozen=True, slots=True)
class ProviderSolution:
    """Exact immutable result returned by one lattice-surgery compiler."""

    epochs: tuple[LatticeSurgeryEpoch, ...]
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        epochs = tuple(self.epochs)
        if not epochs or any(
                not isinstance(value, LatticeSurgeryEpoch) for value in epochs):
            raise TypeError(
                "a provider solution requires typed temporal epochs")
        if tuple(value.index for value in epochs) != tuple(range(len(epochs))):
            raise ValueError(
                "provider-solution epoch indices must be contiguous")
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "payload", _frozen_mapping(self.payload))


class LatticeSurgeryCompiler(QECNetworkCompiler):
    """Base class for a versioned device-attached P2 compiler provider.

    Planning freezes an exact external lattice-surgery artifact.  Materializing
    that artifact produces an ordinary, inspectable P2 protocol network.  A
    separate physical-projector capability owns the later P2-to-P3 step.
    """

    __slots__ = ("_plugin", "_version", "_name", "_pipeline", "_key")
    _CONFIGURATION_FIELDS = frozenset(__slots__)

    def __setattr__(self, name, value) -> None:
        if name in self._CONFIGURATION_FIELDS and hasattr(self, name):
            raise AttributeError(
                f"lattice-surgery compiler configuration {name!r} is "
                "immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name) -> None:
        if name in self._CONFIGURATION_FIELDS:
            raise AttributeError(
                f"lattice-surgery compiler configuration {name!r} is "
                "immutable")
        object.__delattr__(self, name)

    def __init__(self, *, plugin: str, version: str, name: str, pipeline):
        for value, what in (
            (plugin, "plugin"),
            (version, "version"),
            (name, "name"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"lattice-surgery compiler {what} must be nonempty")
        _pipeline_record(pipeline)
        if getattr(pipeline, "output_profile", None) != "p2n":
            raise ValueError(
                "lattice-surgery compiler pipeline must produce P2N")
        self._plugin = plugin
        self._version = version
        self._name = name
        self._pipeline = pipeline
        self._key = f"{plugin}:{name}@{version}"

    @property
    def plugin(self) -> str:
        return self._plugin

    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def key(self) -> str:
        return self._key

    @property
    def qec_lowerings(self) -> tuple[QECLowering, ...]:
        """Module-linked lowerings contributed with this compiler.

        Providers that participate in ordinary P1-to-P2 selection override
        this property.  An empty tuple preserves the explicit research API for
        operation-only problems that have no retained P1 action sites.
        """

        return ()

    def accepts(self, problem: LatticeSurgeryProblem, device: Device) -> bool:
        raise NotImplementedError

    def check_compatibility(
        self,
        request: QECNetworkRequest,
        context: QECNetworkContext,
    ) -> Compatibility:
        """Return a diagnostic compatibility verdict for a network request.

        Legacy providers may continue to implement :meth:`accepts` while they
        migrate.  New providers should override this method so QLX can retain
        precise rejection diagnostics.
        """

        problem = LatticeSurgeryProblem(
            operations=request.operations,
            strategy=scheduling.greedy_asap,
            metadata={"policy": request.policy},
        )
        if not isinstance(context.device, Device):
            raise TypeError(
                "network planning context requires its concrete device")
        return Compatibility(bool(self.accepts(problem, context.device)))

    def architecture_digest(self, device: Device) -> str:
        raise NotImplementedError

    def accepts_pipeline(self, pipeline) -> bool:
        """Whether materialization implements this exact P2N recipe."""

        return pipeline == self.pipeline

    def solve(
        self,
        problem: LatticeSurgeryProblem,
        device: Device,
        *,
        strategy: str,
    ) -> ProviderSolution:
        """Use core temporal mapping unless a provider overrides globally."""

        return pack_epochs(
            problem,
            planner=self,
            device=device,
            strategy=strategy,
        )

    def plan_network(
        self,
        request: QECNetworkRequest,
        context: QECNetworkContext,
    ) -> QECNetworkPlan:
        """Plan the complete selected network.

        Canonical ``qec()`` compilation never infers a provider schedule in
        core QLX.  A network compiler must return every epoch, ownership
        exclusion, and temporal-resource claim.  The operation-only
        ``solve()`` API remains available as an explicit compatibility path.
        """

        del request, context
        raise NotImplementedError(
            f"network compiler {self.key!r} must implement plan_network()")

    def emit_region(self, region, output) -> None:
        """Emit one planned region through a prepared core-owned builder."""

        raise NotImplementedError

    def initial_epoch_state(
        self,
        problem: LatticeSurgeryProblem,
        device: Device,
    ) -> Any:
        """Return provider-private state at the first temporal epoch."""

        return None

    def plan_epoch(
        self,
        context: EpochPlanningContext,
        operations: tuple[LatticeSurgeryOperation, ...],
        device: Device,
        *,
        state: Any,
    ) -> EpochCandidate | EpochInfeasible:
        """Prove or reject one simultaneous batch on this architecture.

        Implementations are deterministic and side-effect free. State changes
        are proposed only through ``EpochCandidate.next_state`` and become
        visible after core QLX commits that candidate.
        """

        raise NotImplementedError(
            f"lattice-surgery compiler {self.key!r} must override solve() or "
            "implement plan_epoch()")

    def finish_epoch_plan(
        self,
        problem: LatticeSurgeryProblem,
        device: Device,
        *,
        epochs: tuple[EpochCandidate, ...],
        state: Any,
    ) -> Mapping[str, Any]:
        """Assemble exact provider payload after core temporal mapping."""

        return {
            "schema":
                "qlx.lattice_surgery.epoch_plan/v1",
            "epochs": [{
                "index": index,
                "operations": list(epoch.operations),
                "provider_digest": epoch.provider_digest,
                "witness": _json_value(epoch.witness),
                "claims": [value.to_dict() for value in epoch.claims],
                "metrics": _json_value(epoch.metrics),
            } for index, epoch in enumerate(epochs)],
        }

    def materialize_p2(
        self,
        plan: "LatticeSurgeryPlan",
        device: Device,
        *,
        pipeline,
        context: QECNetworkContext | None = None,
    ):
        raise NotImplementedError

    def verify_p2_materialization(
        self,
        plan: "LatticeSurgeryPlan",
        device: Device,
        result,
    ) -> None:
        """Verify provider-owned P2 semantics in the returned build."""

        raise NotImplementedError


def _strategy_name(value) -> str:
    if isinstance(value, SchedulingStrategy):
        if not value.supports("lattice_surgery"):
            raise ValueError("temporal lattice-surgery mapping requires a "
                             "lattice-surgery scheduling strategy")
        return str(value)
    if not isinstance(value, str) or not value:
        raise TypeError(
            "temporal lattice-surgery mapping strategy must be a nonempty "
            "string or typed qlx.scheduling strategy")
    return value


def _slot_keys(
        operation: LatticeSurgeryOperation) -> frozenset[tuple[str, int]]:
    return frozenset((slot.space.name, slot.index) for slot in operation.slots)


def pack_epochs(
    problem: LatticeSurgeryProblem,
    *,
    planner: LatticeSurgeryCompiler,
    device: Device,
    strategy: str | SchedulingStrategy | None = None,
) -> ProviderSolution:
    """Pack requests into epochs using core legality and a provider oracle.

    Core QLX owns dependency readiness, logical-slot exclusion, deterministic
    iteration, and immutable epoch summaries. ``planner.plan_epoch`` owns the
    complete architecture-specific feasibility decision and exact witness for
    every candidate batch.
    """

    if not isinstance(problem, LatticeSurgeryProblem):
        raise TypeError("pack_epochs requires a typed lattice-surgery problem")
    if not isinstance(planner, LatticeSurgeryCompiler):
        raise TypeError("pack_epochs planner must be a LatticeSurgeryCompiler")
    if not isinstance(device, Device):
        raise TypeError("pack_epochs requires a typed QLX device")
    strategy_name = (problem.strategy
                     if strategy is None else _strategy_name(strategy))
    if strategy_name != problem.strategy:
        raise ValueError(
            "temporal mapper strategy must match its lattice-surgery problem")
    if strategy_name != str(scheduling.greedy_asap):
        raise ValueError(
            "core lattice-surgery temporal mapping currently supports only "
            "qlx.scheduling.greedy_asap")

    remaining = list(problem.operations)
    completed: set[str] = set()
    accepted: list[EpochCandidate] = []
    state = planner.initial_epoch_state(problem, device)

    while remaining:
        context = EpochPlanningContext(
            problem=problem,
            index=len(accepted),
            completed=frozenset(completed),
            remaining=tuple(remaining),
        )
        ready = tuple(operation for operation in remaining
                      if set(operation.after) <= completed)
        if not ready:
            raise PlacementInfeasible(
                "lattice-surgery dependency graph has no schedulable "
                "operation")

        selected: list[LatticeSurgeryOperation] = []
        selected_slots: set[tuple[str, int]] = set()
        selected_candidate = None
        rejections = []
        for operation in ready:
            keys = _slot_keys(operation)
            if keys & selected_slots:
                continue
            candidate_operations = (*selected, operation)
            proposal = planner.plan_epoch(
                context,
                candidate_operations,
                device,
                state=state,
            )
            if isinstance(proposal, EpochInfeasible):
                rejections.append((operation.name, proposal.reason))
                continue
            if not isinstance(proposal, EpochCandidate):
                raise TypeError(
                    f"lattice-surgery compiler {planner.key!r} plan_epoch() "
                    "must return EpochCandidate or EpochInfeasible")
            expected = tuple(value.name for value in candidate_operations)
            if proposal.operations != expected:
                raise ValueError(
                    f"lattice-surgery compiler {planner.key!r} returned "
                    f"candidate operations {proposal.operations!r}, expected "
                    f"{expected!r}")
            selected = list(candidate_operations)
            selected_slots.update(keys)
            selected_candidate = proposal

        if not selected or selected_candidate is None:
            details = "; ".join(
                f"{name}: {reason}" for name, reason in rejections)
            suffix = f" ({details})" if details else ""
            raise PlacementInfeasible(
                "no ready lattice-surgery operation has a feasible temporal "
                f"placement in epoch {context.index}{suffix}")

        accepted.append(selected_candidate)
        state = selected_candidate.next_state
        selected_names = {value.name for value in selected}
        remaining = [
            value for value in remaining if value.name not in selected_names
        ]
        completed.update(selected_names)

    payload = planner.finish_epoch_plan(
        problem,
        device,
        epochs=tuple(accepted),
        state=state,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"lattice-surgery compiler {planner.key!r} finish_epoch_plan() "
            "must return a mapping")
    return ProviderSolution(
        epochs=tuple(
            LatticeSurgeryEpoch(
                index=index,
                operations=candidate.operations,
                provider_digest=candidate.provider_digest,
                claims=candidate.claims,
            ) for index, candidate in enumerate(accepted)),
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class LatticeSurgeryPlan:
    """Exact provider-bound artifact consumed by
    :func:`cudaq.logical.compile`.

    The source problem is retained so compilation cannot accidentally solve a
    similar-but-different problem.  ``provider_payload`` is the exact
    serializable artifact returned by the selected external compiler.
    """

    problem: LatticeSurgeryProblem
    provider_key: str
    device_name: str
    architecture_digest: str
    materialization_pipeline_digest: str
    epochs: tuple[LatticeSurgeryEpoch, ...]
    provider_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.problem, LatticeSurgeryProblem):
            raise TypeError(
                "lattice-surgery plans require their exact source problem")
        for value, what in (
            (self.provider_key, "provider key"),
            (self.device_name, "device name"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"lattice-surgery plan {what} must be nonempty")
        _require_digest(
            self.architecture_digest,
            what="lattice-surgery plan architecture digest",
        )
        _require_digest(
            self.materialization_pipeline_digest,
            what="lattice-surgery plan materialization-pipeline digest",
        )
        epochs = tuple(self.epochs)
        if not epochs or any(
                not isinstance(value, LatticeSurgeryEpoch) for value in epochs):
            raise TypeError(
                "lattice-surgery plans require typed temporal epochs")
        if tuple(value.index for value in epochs) != tuple(range(len(epochs))):
            raise ValueError("lattice-surgery plan epochs must be contiguous")
        scheduled = tuple(name for epoch in epochs for name in epoch.operations)
        expected = tuple(value.name for value in self.problem.operations)
        if sorted(scheduled) != sorted(expected):
            raise ValueError(
                "lattice-surgery plan epochs must schedule every operation "
                "exactly once")
        epoch_for = {
            name: epoch.index for epoch in epochs for name in epoch.operations
        }
        operations = {value.name: value for value in self.problem.operations}
        for name, operation in operations.items():
            for dependency in operation.after:
                if epoch_for[dependency] >= epoch_for[name]:
                    raise ValueError(
                        f"lattice-surgery operation {name!r} must be in a "
                        f"strictly later epoch than dependency {dependency!r}")
        for epoch in epochs:
            occupied = set()
            for name in epoch.operations:
                keys = {(slot.space.name, slot.index)
                        for slot in operations[name].slots}
                overlap = keys & occupied
                if overlap:
                    raise ValueError(
                        f"lattice-surgery epoch {epoch.index} reuses logical "
                        f"slot(s) {sorted(overlap)!r}")
                occupied.update(keys)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(
            self,
            "provider_payload",
            _frozen_mapping(self.provider_payload),
        )

    @property
    def strategy(self) -> str:
        return self.problem.strategy

    @property
    def makespan_epochs(self) -> int:
        return len(self.epochs)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "qlx.lattice_surgery.plan/v6",
            "problem": self.problem.to_dict(),
            "problem_digest": self.problem.digest,
            "provider_key": self.provider_key,
            "device_name": self.device_name,
            "architecture_digest": self.architecture_digest,
            "materialization_pipeline_digest":
                (self.materialization_pipeline_digest),
            "epochs": [value.to_dict() for value in self.epochs],
            "provider_payload": _json_value(self.provider_payload),
        }

    def to_json(self) -> str:
        return _mapping_module().canonical_json({
            **self.to_dict(),
            "digest": self.digest,
        })

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, value, *, device: Device) -> "LatticeSurgeryPlan":
        value = _record(
            value,
            what="lattice-surgery plan",
            keys=(
                "schema",
                "problem",
                "problem_digest",
                "provider_key",
                "device_name",
                "architecture_digest",
                "materialization_pipeline_digest",
                "epochs",
                "provider_payload",
            ),
        )
        if value["schema"] != "qlx.lattice_surgery.plan/v6":
            raise ValueError("unsupported lattice-surgery plan schema")
        if value["device_name"] != device.name:
            raise _mapping_module().MappingVerificationError(
                "lattice-surgery plan device name mismatch")
        source = LatticeSurgeryProblem.from_dict(
            value["problem"],
            device=device,
        )
        problem_digest = value["problem_digest"]
        if (not isinstance(problem_digest, str) or
                problem_digest != source.digest):
            raise _mapping_module().MappingVerificationError(
                "lattice-surgery plan problem digest mismatch")
        epochs = []
        for index, raw in enumerate(
                _array(
                    value["epochs"],
                    what="lattice-surgery plan epochs",
                )):
            raw = _record(
                raw,
                what=f"lattice-surgery plan epoch {index}",
                keys=("index", "operations", "provider_digest", "claims"),
            )
            operations = _array(
                raw["operations"],
                what=f"lattice-surgery plan epoch {index} operations",
            )
            if any(not isinstance(item, str) or not item
                   for item in operations):
                raise TypeError(
                    "lattice-surgery epoch operation names must be strings")
            epochs.append(
                LatticeSurgeryEpoch(
                    index=raw["index"],
                    operations=tuple(operations),
                    provider_digest=raw["provider_digest"],
                    claims=tuple(
                        TemporalResourceClaim.from_dict(item)
                        for item in _array(
                            raw["claims"],
                            what=(f"lattice-surgery plan epoch {index} claims"),
                        )),
                ))
        provider_payload = value["provider_payload"]
        if not isinstance(provider_payload, dict):
            raise TypeError(
                "lattice-surgery provider payload must be a JSON object")
        return cls(
            problem=source,
            provider_key=value["provider_key"],
            device_name=value["device_name"],
            architecture_digest=value["architecture_digest"],
            materialization_pipeline_digest=(
                value["materialization_pipeline_digest"]),
            epochs=tuple(epochs),
            provider_payload=provider_payload,
        )

    @classmethod
    def from_json(cls, text: str, *, device: Device):
        try:
            payload = _strict_json_loads(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid lattice-surgery plan JSON") from exc
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ValueError("lattice-surgery plan JSON requires a digest")
        claimed = payload.pop("digest")
        if not isinstance(claimed, str) or claimed != _digest(payload):
            raise _mapping_module().MappingVerificationError(
                "lattice-surgery plan digest mismatch")
        return cls.from_dict(payload, device=device)

    @classmethod
    def load(cls, path, *, device: Device):
        return cls.from_json(
            Path(path).read_text(encoding="utf-8"),
            device=device,
        )


def request(
        product: PauliProduct,
        *,
        name: str | None = None,
        after: Iterable[LatticeSurgeryOperation] = (),
) -> ProductMeasurement:
    """Create one typed logical product-measurement operation.

    Compose the product with the ordinary QLX Pauli surface, for example
    ``cudaq.logical.Z(compute[0]) @ cudaq.logical.Z(compute[1])``.
    """

    return ProductMeasurement(product, name=name, after=after)


def _attribute_text(attribute) -> str:
    return str(getattr(attribute, "value", attribute)).strip('@"')


def _integer_attribute(attribute, *, what: str) -> int:
    try:
        return int(str(attribute).split()[0])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be an integer attribute") from exc


def _single_block(operation, *, what: str):
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        raise ValueError(f"{what} must have one region and one block")
    return operation.regions[0].blocks[0]


def _dialect_key(attribute, *, prefix: str, what: str) -> str:
    text = str(attribute)
    opening = f"#{prefix}<"
    if not text.startswith(opening) or not text.endswith(">"):
        raise ValueError(f"{what} has malformed attribute {text!r}")
    value = text[len(opening):-1].strip('"')
    if not value:
        raise ValueError(f"{what} must be nonempty")
    return value


def _p1_machine_contract(build, machine_name: str) -> Mapping[str, Any]:
    """Recover the complete logical-machine contract from typed P1 IR."""

    try:
        domain = build.definitions[machine_name]
    except KeyError as exc:
        raise ValueError(f"placed-program P1 artifact omitted logical domain "
                         f"@{machine_name}") from exc
    if domain.kind != "lvm.domain":
        raise ValueError(
            f"placed-program P1 logical domain @{machine_name} has kind "
            f"{domain.kind!r}")
    block = _single_block(domain.op, what="placed-program P1 logical domain")
    spaces = []
    streams = []
    channels = []
    for view in block.operations:
        operation = view.operation
        attributes = operation.attributes
        if operation.name == "lvm.space":
            spaces.append({
                "name":
                    _attribute_text(attributes["sym_name"]),
                "capacity": (_integer_attribute(
                    attributes["capacity"],
                    what="placed-program P1 space capacity",
                ) if "capacity" in attributes else None),
                "capabilities":
                    tuple(
                        _dialect_key(
                            value,
                            prefix="lvm.capability",
                            what="placed-program P1 space capability",
                        ) for value in attributes["capabilities"]),
                "tags":
                    tuple(
                        _attribute_text(value) for value in attributes["tags"])
                    if "tags" in attributes else (),
            })
        elif operation.name == "lvm.stream":
            streams.append({
                "name":
                    _attribute_text(attributes["sym_name"]),
                "produces":
                    _attribute_text(attributes["produces"]),
                "buffer_size": (_integer_attribute(
                    attributes["capacity"],
                    what="placed-program P1 stream capacity",
                ) if "capacity" in attributes else None),
            })
        elif operation.name == "lvm.channel":
            channels.append({
                "name":
                    _attribute_text(attributes["sym_name"]),
                "source":
                    _attribute_text(attributes["from"]),
                "destination":
                    _attribute_text(attributes["to"]),
                "capabilities":
                    tuple(
                        _dialect_key(
                            value,
                            prefix="lvm.capability",
                            what="placed-program P1 channel capability",
                        ) for value in attributes["capabilities"]),
                "direction":
                    _attribute_text(attributes["direction"]),
                "capacity": (_integer_attribute(
                    attributes["capacity"],
                    what="placed-program P1 channel capacity",
                ) if "capacity" in attributes else None),
            })
    return {
        "name": machine_name,
        "spaces": tuple(spaces),
        "streams": tuple(streams),
        "channels": tuple(channels),
    }


def _require_p1_device_contract(build, placement, device: Device) -> None:
    """Authenticate P1 placement against the supplied live logical machine."""

    from ...architecture.placement import _machine_contract

    retained = _p1_machine_contract(build, placement.machine)
    live = _machine_contract(device.logical)
    if retained != live:
        raise _mapping_module().MappingVerificationError(
            "placed-program P1 logical machine contract does not match the "
            "supplied device")


def _encoded_y_injection_from_build(
    build,
    *,
    program_name: str,
    operands: Iterable[SpaceSlot],
    after: Iterable[LatticeSurgeryOperation],
) -> EncodedYInjection:
    """Recognize and retain one verified P0 Litinski injection artifact."""

    slots = tuple(
        _require_slot(value, what="lattice-surgery program operand")
        for value in operands)
    if build.stage != P0 or not build.verify():
        raise ValueError(
            "lattice-surgery program extraction requires a verified P0 build")
    try:
        root = build.definitions[program_name].op
    except KeyError as exc:
        raise ValueError(
            f"lattice-surgery program @{program_name} is absent from its "
            "compiled P0 build") from exc
    block = _single_block(root, what="lattice-surgery source program")
    arguments = tuple(block.arguments)
    if len(arguments) != len(slots):
        raise ValueError(
            "lattice-surgery source program ABI does not match its typed "
            "operand binding")
    slot_for = dict(zip(arguments, slots))
    operations = tuple(value.operation for value in block.operations)
    if tuple(value.name for value in operations) != (
            "qlx.instrument",
            "qlx.measure",
            "qlx.xor",
            "cflow.if",
            "qlx.return",
    ):
        raise ValueError(
            "encoded-Y injection authoring must be exactly MZZ, "
            "destructive MX, MZZ xor MX, conditional Z, and return")

    mzz, mx, parity, correction, returned = operations
    if _attribute_text(mzz.attributes["instrument"]) != "#qlx.instrument<mpp>":
        raise ValueError("encoded-Y injection must begin with qlx.mpp")
    parameters = mzz.attributes["parameters"]
    arity = len(mzz.operands)
    if (arity != 2 or len(mzz.results) != 3 or _integer_attribute(
            parameters["sign"],
            what="encoded-Y injection MPP sign",
    ) != 1 or _integer_attribute(
            parameters["x_mask"],
            what="encoded-Y injection MPP X mask",
    ) != 0 or _integer_attribute(
            parameters["z_mask"],
            what="encoded-Y injection MPP Z mask",
    ) != 0b11):
        raise ValueError(
            "encoded-Y injection first instruction must be positive M_ZZ")
    try:
        target, factory = tuple(slot_for[value] for value in mzz.operands)
    except KeyError as exc:
        raise ValueError(
            "encoded-Y injection M_ZZ must consume the program operands"
        ) from exc
    slot_for[mzz.results[0]] = target
    slot_for[mzz.results[1]] = factory

    if (len(mx.operands) != 1 or len(mx.results) != 1 or
            mx.operands[0] != mzz.results[1] or
            _attribute_text(mx.attributes["basis"]) != "X"):
        raise ValueError(
            "encoded-Y injection second instruction must be destructive "
            "qlx.measure_x of the encoded-|Y> operand")
    if (len(parity.operands) != 2 or len(parity.results) != 1 or
            set(parity.operands) != {mzz.results[2], mx.results[0]}):
        raise ValueError(
            "encoded-Y injection correction predicate must be M_ZZ xor M_X")
    if len(correction.operands) != 1 or correction.operands[0] != parity.result:
        raise ValueError(
            "encoded-Y injection conditional must consume M_ZZ xor M_X")
    if (len(correction.regions) != 2 or len(correction.results) != 1 or
            any(len(region.blocks) != 1 for region in correction.regions)):
        raise ValueError(
            "encoded-Y injection correction must be one structured branch")
    then_block = correction.regions[0].blocks[0]
    else_block = correction.regions[1].blocks[0]
    then_ops = tuple(value.operation for value in then_block.operations)
    else_ops = tuple(value.operation for value in else_block.operations)
    if (tuple(value.name for value in then_ops) != ("qlx.apply", "cflow.yield")
            or tuple(value.name for value in else_ops) != ("cflow.yield",) or
            _attribute_text(
                then_ops[0].attributes["action"]) != "#qlx.action<z>" or
            tuple(then_ops[0].operands) != (mzz.results[0],) or
            tuple(then_ops[1].operands) != (then_ops[0].result,) or
            tuple(else_ops[0].operands) != (mzz.results[0],) or
            tuple(returned.operands) != (correction.result,)):
        raise ValueError(
            "encoded-Y injection branch must apply Z to the data operand "
            "exactly when M_ZZ xor M_X is true")

    return EncodedYInjection._from_verified_source(
        target=target,
        factory=factory,
        name=program_name,
        source_artifact=build.serialize().hex(),
        after=after,
    )


def _encoded_y_injection_from_program(
    definition: ProgramDefinition,
    *,
    operands: Iterable[SpaceSlot],
    after: Iterable[LatticeSurgeryOperation],
) -> EncodedYInjection:
    """Recognize the typed Litinski injection program without text parsing."""

    if definition.kind != "program" or definition.stage != P0:
        raise TypeError(
            "lattice-surgery program extraction requires a device-free "
            "@cudaq.logical.program")
    slots = tuple(
        _require_slot(value, what="lattice-surgery program operand")
        for value in operands)
    if len(slots) != len(definition.signature.parameters):
        raise ValueError(
            f"lattice-surgery program @{definition.name} has "
            f"{len(definition.signature.parameters)} operands, but "
            f"{len(slots)} typed slots were supplied")
    return _encoded_y_injection_from_build(
        definition.materialize(),
        program_name=definition.name,
        operands=slots,
        after=after,
    )


def _encoded_y_injection_from_artifact(
    source_artifact: str,
    *,
    operands: Iterable[SpaceSlot],
    after: Iterable[LatticeSurgeryOperation],
) -> EncodedYInjection:
    if not isinstance(source_artifact, str) or not source_artifact:
        raise TypeError(
            "encoded-Y injection source artifact must be a nonempty "
            "canonical replay bundle")
    try:
        payload = bytes.fromhex(source_artifact)
    except ValueError as exc:
        raise ValueError(
            "encoded-Y injection source artifact must be hexadecimal") from exc
    from ...compiler import Build

    try:
        build = Build.replay(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "encoded-Y injection source artifact is not a valid canonical "
            "QLX replay bundle") from exc
    if build.serialize() != payload:
        raise ValueError("encoded-Y injection source artifact is not canonical")
    return _encoded_y_injection_from_build(
        build,
        program_name=build.root.symbol,
        operands=operands,
        after=after,
    )


def _wrapped_attribute(attribute, *, prefix: str, what: str) -> str:
    text = _attribute_text(attribute)
    opening = f"#{prefix}<"
    if text.startswith(opening) and text.endswith(">"):
        text = text[len(opening):-1]
    if not text:
        raise ValueError(f"{what} has unsupported attribute {text!r}")
    return text.upper()


def _placed_program_from_build(
    build,
    *,
    device: Device | None = None,
    slots: Iterable[SpaceSlot] = (),
) -> tuple[LatticeSurgeryProgram, tuple[ProductMeasurement, ...]]:
    """Extract a closed straight-line logical program from verified P1.

    P1 remains the authority for exact logical placement.  This function
    validates the retained P0 SSA graph and converts only the supported,
    provider-neutral subset into immutable typed records.  Unsupported control
    flow and ambiguous owner lineage fail closed.
    """

    if build.stage not in {P1, P2
                          } or build.placement is None or not build.verify():
        raise ValueError(
            "placed lattice-surgery program extraction requires a verified "
            "P1/P2 build with an exact placement witness")
    payload = build.serialize()
    from ...compiler import Build

    replayed = Build.replay(payload)
    if replayed.serialize() != payload:
        raise ValueError("placed-program source artifact is not canonical")
    placement = build.placement
    try:
        source = build.definitions[placement.input_p0].op
    except KeyError as exc:
        raise ValueError(
            "placed-program P1 artifact omitted its retained P0 program"
        ) from exc
    block = _single_block(source, what="placed lattice-surgery source program")
    if tuple(block.arguments):
        raise ValueError(
            "placed lattice-surgery programs are currently closed programs; "
            "prepare logical inputs explicitly")

    slot_lookup = {
        (slot.space.name, slot.index):
            _require_slot(slot, what="placed-program replay slot")
        for slot in slots
    }
    if device is not None:
        if not isinstance(device, Device):
            raise TypeError(
                "placed-program extraction device must be a typed cudaq.logical.Device"
            )
        _require_p1_device_contract(build, placement, device)
        slot_lookup = {
            (space.name, index): space[index] for space in device.logical.spaces
            if space.name is not None and space.capacity is not None
            for index in range(space.capacity)
        }
    if not slot_lookup:
        raise ValueError(
            "placed-program extraction requires the live device or its exact "
            "typed slots")

    by_source = {}
    for binding in placement.bindings:
        if binding.binding_kind != "local":
            raise ValueError(
                "placed lattice-surgery programs currently require exact "
                "local P1 bindings")
        if binding.source_allocation is None:
            raise ValueError(
                "closed placed programs require allocation-backed P1 bindings")
        try:
            resolved = slot_lookup[(binding.space, binding.slot)]
        except KeyError as exc:
            raise ValueError(
                f"placed-program binding @{binding.space}[{binding.slot}] is "
                "absent from the live device") from exc
        key = (binding.source_allocation, tuple(binding.source_path))
        if key in by_source:
            raise ValueError("placed-program source bindings are ambiguous")
        by_source[key] = resolved

    instructions = []
    measurements: list[ProductMeasurement] = []
    owner_by_value = {}
    dependencies_by_value = {}
    record_by_value = {}
    record_sources = {}
    counters: dict[str, int] = {}
    barrier_dependencies: set[str] = set()
    open_mpp_segment: list[str] = []

    def fresh(prefix: str) -> str:
        index = counters.get(prefix, 0)
        counters[prefix] = index + 1
        return f"{prefix}{index}"

    def close_mpp_segment() -> None:
        """Make every MPP in the next run wait for the complete prior run."""

        if open_mpp_segment:
            barrier_dependencies.update(open_mpp_segment)
            open_mpp_segment.clear()

    def owner(value, *, what: str) -> SpaceSlot:
        try:
            return owner_by_value[value]
        except KeyError as exc:
            raise ValueError(
                f"{what} has unresolved P1 logical ownership") from exc

    def record(value, *, what: str) -> str:
        try:
            return record_by_value[value]
        except KeyError as exc:
            raise ValueError(f"{what} uses an unresolved record") from exc

    operations = tuple(value.operation for value in block.operations)
    if not operations or operations[-1].name != "qlx.return":
        raise ValueError("placed lattice-surgery programs must end in return")
    for operation in operations:
        name = operation.name
        if name == "qlx.prepare":
            try:
                allocation = _integer_attribute(
                    operation.attributes["allocation"],
                    what="placed-program allocation",
                )
                value_index = _integer_attribute(
                    operation.attributes["value_index"],
                    what="placed-program allocation value index",
                )
                state = _attribute_text(operation.attributes["state"])
                slot = by_source[(allocation, (value_index,))]
            except KeyError as exc:
                raise ValueError(
                    "placed-program preparation does not resolve through its "
                    "P1 placement witness") from exc
            if state not in {"zero", "plus"}:
                raise ValueError(
                    "placed lattice-surgery preparation supports only zero "
                    "and plus")
            if len(operation.results) != 1:
                raise ValueError(
                    "placed-program preparation must produce one logical qubit")
            owner_by_value[operation.result] = slot
            dependencies_by_value[operation.result] = frozenset()
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.PREPARE,
                    fresh("prepare"),
                    slots=(slot,),
                    state=state,
                ))
            close_mpp_segment()
            continue

        if name == "qlx.apply":
            if len(operation.operands) != 1 or len(operation.results) != 1:
                raise ValueError(
                    "placed-program Pauli operations must be unary")
            pauli = _wrapped_attribute(
                operation.attributes["action"],
                prefix="qlx.action",
                what="placed-program action",
            )
            if pauli not in {"X", "Z"}:
                raise ValueError(
                    "placed lattice-surgery programs currently support only "
                    "logical X and Z")
            source_value = operation.operands[0]
            slot = owner(source_value, what="placed-program Pauli")
            owner_by_value[operation.result] = slot
            dependencies_by_value[operation.result] = (
                dependencies_by_value[source_value])
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.PAULI,
                    fresh("pauli"),
                    slots=(slot,),
                    paulis=(pauli,),
                ))
            close_mpp_segment()
            continue

        if name == "qlx.instrument":
            instrument = _wrapped_attribute(
                operation.attributes["instrument"],
                prefix="qlx.instrument",
                what="placed-program instrument",
            )
            if instrument != "MPP":
                raise ValueError(
                    "placed lattice-surgery programs support only MPP "
                    "instruments")
            operands = tuple(operation.operands)
            arity = len(operands)
            if len(operation.results) != arity + 1:
                raise ValueError("placed-program MPP result arity is invalid")
            parameters = operation.attributes["parameters"]
            sign = _integer_attribute(parameters["sign"],
                                      what="placed-program MPP sign")
            x_mask = _integer_attribute(parameters["x_mask"],
                                        what="placed-program MPP X mask")
            z_mask = _integer_attribute(parameters["z_mask"],
                                        what="placed-program MPP Z mask")
            if sign != 1:
                raise ValueError(
                    "placed lattice-surgery programs require positive MPPs")
            mpp_slots = tuple(
                owner(value, what="placed-program MPP") for value in operands)
            paulis = []
            for index in range(arity):
                x = bool(x_mask & (1 << index))
                z = bool(z_mask & (1 << index))
                if not x and not z:
                    raise ValueError(
                        "placed-program MPP cannot contain identity operands")
                paulis.append("Y" if x and z else "X" if x else "Z")
            dependency_names = set().union(
                *(dependencies_by_value[value] for value in operands))
            dependency_names.update(barrier_dependencies)
            after = tuple(value for value in measurements
                          if value.name in dependency_names)
            measurement_name = fresh("mpp")
            measurement = ProductMeasurement(
                PauliProduct(
                    tuple(
                        PauliFactor(slot, pauli)
                        for slot, pauli in zip(mpp_slots, paulis))),
                name=measurement_name,
                after=after,
            )
            measurements.append(measurement)
            next_dependencies = frozenset((*dependency_names, measurement_name))
            for source_value, result_value in zip(
                    operands,
                    tuple(operation.results)[:arity]):
                owner_by_value[result_value] = owner_by_value[source_value]
                dependencies_by_value[result_value] = next_dependencies
            outcome = operation.results[arity]
            record_by_value[outcome] = measurement_name
            record_sources[measurement_name] = next_dependencies
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.MEASURE_PRODUCT,
                    measurement_name,
                    slots=mpp_slots,
                    paulis=tuple(paulis),
                    outputs=(measurement_name,),
                ))
            open_mpp_segment.append(measurement_name)
            continue

        if name == "qlx.measure":
            if len(operation.operands) != 1 or len(operation.results) != 1:
                raise ValueError(
                    "placed-program logical measurement must be unary")
            basis = _wrapped_attribute(
                operation.attributes["basis"],
                prefix="qlx.pauli",
                what="placed-program measurement basis",
            )
            if basis not in {"X", "Z"}:
                raise ValueError(
                    "placed lattice-surgery readout supports only X and Z")
            source_value = operation.operands[0]
            slot = owner(source_value, what="placed-program readout")
            result_name = fresh("measure")
            record_by_value[operation.result] = result_name
            record_sources[result_name] = dependencies_by_value[source_value]
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.MEASURE,
                    result_name,
                    slots=(slot,),
                    outputs=(result_name,),
                    basis=basis,
                ))
            close_mpp_segment()
            continue

        if name == "qlx.xor":
            if len(operation.operands) != 2 or len(operation.results) != 1:
                raise ValueError("placed-program XOR must be binary")
            inputs = tuple(
                record(value, what="placed-program XOR")
                for value in operation.operands)
            result_name = fresh("xor")
            record_by_value[operation.result] = result_name
            record_sources[result_name] = frozenset().union(
                *(record_sources[value] for value in inputs))
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.XOR,
                    result_name,
                    inputs=inputs,
                    outputs=(result_name,),
                ))
            close_mpp_segment()
            continue

        if name == "cflow.if":
            if (len(operation.operands) != 1 or len(operation.results) != 1 or
                    len(operation.regions) != 2 or any(
                        len(region.blocks) != 1
                        for region in operation.regions)):
                raise ValueError(
                    "placed-program feedback must be one unary structured if")
            condition = record(operation.operands[0],
                               what="placed-program feedback")
            then_ops = tuple(
                value.operation
                for value in operation.regions[0].blocks[0].operations)
            else_ops = tuple(
                value.operation
                for value in operation.regions[1].blocks[0].operations)
            if (tuple(value.name for value in then_ops)
                    != ("qlx.apply", "cflow.yield") or tuple(
                        value.name for value in else_ops) != ("cflow.yield",)):
                raise ValueError(
                    "placed-program feedback must condition one logical Pauli")
            applied, then_yield = then_ops
            else_yield, = else_ops
            if (len(applied.operands) != 1 or len(applied.results) != 1 or
                    tuple(then_yield.operands) != (applied.result,) or
                    tuple(else_yield.operands) != (applied.operands[0],)):
                raise ValueError(
                    "placed-program feedback branch has invalid logical "
                    "lineage")
            pauli = _wrapped_attribute(
                applied.attributes["action"],
                prefix="qlx.action",
                what="placed-program feedback action",
            )
            if pauli not in {"X", "Z"}:
                raise ValueError(
                    "placed-program feedback supports only X and Z")
            source_value = applied.operands[0]
            slot = owner(source_value, what="placed-program feedback")
            owner_by_value[operation.result] = slot
            dependencies_by_value[operation.result] = frozenset((
                *dependencies_by_value[source_value],
                *record_sources[condition],
            ))
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.CONDITIONAL_PAULI,
                    fresh("conditional_pauli"),
                    slots=(slot,),
                    paulis=(pauli,),
                    inputs=(condition,),
                ))
            close_mpp_segment()
            continue

        if name == "qlx.return":
            if operation is not operations[-1]:
                raise ValueError("placed-program return must be terminal")
            returned = tuple(
                record(value, what="placed-program return")
                for value in operation.operands)
            instructions.append(
                ProgramInstruction(
                    ProgramInstructionKind.RETURN,
                    "return",
                    inputs=returned,
                ))
            continue

        raise ValueError(
            f"placed lattice-surgery programs do not support {name!r}")

    program = LatticeSurgeryProgram._from_verified_source(
        name=placement.input_p0,
        source_artifact=payload.hex(),
        instructions=instructions,
    )
    if not measurements:
        raise ValueError(
            "placed lattice-surgery programs require at least one MPP")
    return program, tuple(measurements)


def _placed_program_from_artifact(
    source_artifact: str,
    *,
    device: Device | None = None,
    slots: Iterable[SpaceSlot] = (),
) -> tuple[LatticeSurgeryProgram, tuple[ProductMeasurement, ...]]:
    if not isinstance(source_artifact, str) or not source_artifact:
        raise TypeError(
            "placed-program source artifact must be a canonical replay bundle")
    try:
        payload = bytes.fromhex(source_artifact)
    except ValueError as exc:
        raise ValueError(
            "placed-program source artifact must be hexadecimal") from exc
    from ...compiler import Build

    try:
        build = Build.replay(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "placed-program source artifact is not a canonical QLX replay "
            "bundle") from exc
    if build.serialize() != payload:
        raise ValueError("placed-program source artifact is not canonical")
    return _placed_program_from_build(build, device=device, slots=slots)


def _source_p1_build(problem_value: LatticeSurgeryProblem):
    """Replay the exact canonical P1 artifact retained by a closed problem."""

    program = problem_value.program
    if program is None:
        return None
    try:
        payload = bytes.fromhex(program.source_artifact)
    except ValueError as exc:
        raise ValueError(
            "placed-program source artifact must be hexadecimal") from exc
    from ...compiler import Build

    source = Build.replay(payload)
    if (source.serialize() != payload or source.profile != "p1" or
            not source.verify()):
        raise ValueError(
            "lattice-surgery problem does not retain one canonical verified P1 build"
        )
    return source


def _network_context(plan, device, compiler):
    source = _source_p1_build(plan.problem)
    if source is None:
        return None
    from ...compiler.build import _qec_selection_sha256
    from ...compiler.qec_lower import _network_qec_selection

    selected = _network_qec_selection(
        source,
        device=device,
        expected_compiler=compiler,
    )
    if selected is None:
        raise ValueError(
            "closed lattice-surgery plan has no selected network QECLowering")
    # The operation-only compatibility route retains an ordinary QEC witness,
    # not the canonical whole-network request/plan contract.  Clearing this
    # marker prevents its provider-owned P2 from being misclassified as a
    # canonical network Build during replay.
    compatibility_selection = replace(
        selected.witness,
        network_manifest_sha256=None,
    )
    return QECNetworkContext(
        source=source,
        lowering=selected.lowering,
        selection=compatibility_selection,
        selection_digest=_qec_selection_sha256(compatibility_selection),
        device=device,
    )


def problem(
    *operations: LatticeSurgeryOperation | ProgramDefinition | Any,
    operands: Iterable[SpaceSlot] = (),
    after: Iterable[LatticeSurgeryOperation] = (),
    device: Device | None = None,
    strategy: str | SchedulingStrategy = scheduling.greedy_asap,
    metadata: Mapping[str, Any] | None = None,
) -> LatticeSurgeryProblem:
    """Create a mapping problem from typed operations or one ordinary program.

    ``operands=`` binds a device-free program's logical ABI positionally to
    concrete P1 slots.  The extractor walks typed MLIR operations and fails
    closed; it never searches or parses printed MLIR.
    """

    from ...compiler import Build

    placed_program = None
    if len(operations) == 1 and isinstance(operations[0], Build):
        if tuple(operands) or tuple(after):
            raise TypeError(
                "a placed P1 build already owns its operand bindings and "
                "dependencies")
        if device is None:
            raise TypeError(
                "problem(P1_build) requires device= to resolve exact typed "
                "slots")
        placed_program, operations = _placed_program_from_build(operations[0],
                                                                device=device)
    elif len(operations) == 1 and isinstance(operations[0], ProgramDefinition):
        if device is not None:
            raise TypeError(
                "device= is available only when problem() receives one P1 "
                "cudaq.logical.Build")
        operations = (_encoded_y_injection_from_program(
            operations[0],
            operands=operands,
            after=after,
        ),)
    elif tuple(operands) or tuple(after) or device is not None:
        raise TypeError(
            "operands=/after= are available only for one "
            "@cudaq.logical.program; "
            "device= is available only for one P1 cudaq.logical.Build")
    return LatticeSurgeryProblem(
        operations=tuple(operations),
        program=placed_program,
        strategy=strategy,
        metadata=dict(metadata or {}),
    )


def _device_compiler(
    problem_value: LatticeSurgeryProblem,
    device: Device,
    *,
    provider_key: str | None = None,
):
    if not isinstance(device, Device):
        raise TypeError(
            "lattice-surgery compilation requires a typed cudaq.logical.Device")
    eligible = tuple(value for value in device.compilers
                     if isinstance(value, LatticeSurgeryCompiler) and
                     (provider_key is None or value.key == provider_key))
    candidates = []
    for value in eligible:
        verdict = value.accepts(problem_value, device)
        if type(verdict) is not bool:
            raise TypeError(
                f"lattice-surgery compiler {value.key!r} accepts() must "
                "return bool")
        if verdict:
            candidates.append(value)
    candidates = tuple(candidates)
    if not candidates:
        # A sole provider can surface its exact architecture incompatibility
        # without creating ambiguity with another feasible implementation.
        if len(eligible) == 1:
            eligible[0].architecture_digest(device)
        requested = ("" if provider_key is None else f" {provider_key!r}")
        raise LookupError(
            f"device @{device.name} has no compatible lattice-surgery "
            f"compiler{requested}")
    if len(candidates) != 1:
        raise LookupError("lattice-surgery compiler selection is ambiguous: " +
                          ", ".join(value.key for value in candidates))
    return candidates[0]


def _bound_device_compiler(device: Device, provider_key: str):
    """Resolve an already-solved plan's provider without replanning policy."""

    if not isinstance(device, Device):
        raise TypeError(
            "lattice-surgery compilation requires a typed cudaq.logical.Device")
    candidates = tuple(value for value in device.compilers
                       if isinstance(value, LatticeSurgeryCompiler) and
                       value.key == provider_key)
    if not candidates:
        raise LookupError(
            f"device @{device.name} has no lattice-surgery compiler "
            f"{provider_key!r} bound by the solved plan")
    if len(candidates) != 1:
        raise LookupError(
            "solved lattice-surgery compiler selection is ambiguous: " +
            ", ".join(value.key for value in candidates))
    return candidates[0]


def _validate_problem_slots(
    problem_value: LatticeSurgeryProblem,
    device: Device,
) -> None:
    spaces = {value.name: value for value in device.logical.spaces}
    for operation in problem_value.operations:
        for slot in operation.slots:
            name = slot.space.name
            target = spaces.get(name)
            if target is None:
                raise ValueError(
                    f"lattice-surgery slot @{name}[{slot.index}] is absent "
                    f"from device @{device.name}")
            if target is not slot.space:
                raise ValueError(
                    f"lattice-surgery slot @{name}[{slot.index}] does not "
                    f"belong to device @{device.name}")
            if (target.capacity is not None and slot.index >= target.capacity):
                raise ValueError(
                    f"lattice-surgery slot @{name}[{slot.index}] exceeds "
                    f"device capacity {target.capacity}")


def solve(
    problem_value: LatticeSurgeryProblem,
    *,
    device: Device,
    compiler: LatticeSurgeryCompiler | None = None,
) -> LatticeSurgeryPlan:
    """Solve once and bind the exact provider artifact to the device."""

    if not isinstance(problem_value, LatticeSurgeryProblem):
        raise TypeError("lattice_surgery.solve requires a typed problem")
    _validate_problem_slots(problem_value, device)
    if compiler is None:
        compiler = _device_compiler(problem_value, device)
    elif (not isinstance(compiler, LatticeSurgeryCompiler) or
          compiler not in device.compilers):
        raise ValueError("compiler= must be a compatible compiler attached "
                         "to device")
    else:
        verdict = compiler.accepts(problem_value, device)
        if type(verdict) is not bool:
            raise TypeError(
                f"lattice-surgery compiler {compiler.key!r} accepts() must "
                "return bool")
        if not verdict:
            raise ValueError(
                "compiler= must be a compatible compiler attached to device")
    architecture_digest = compiler.architecture_digest(device)
    solution = compiler.solve(
        problem_value,
        device,
        strategy=problem_value.strategy,
    )
    if not isinstance(solution, ProviderSolution):
        raise TypeError(f"lattice-surgery compiler {compiler.key!r} returned "
                        f"{type(solution).__name__}, expected ProviderSolution")
    return LatticeSurgeryPlan(
        problem=problem_value,
        provider_key=compiler.key,
        device_name=device.name,
        architecture_digest=architecture_digest,
        materialization_pipeline_digest=_pipeline_digest(compiler.pipeline),
        epochs=solution.epochs,
        provider_payload=solution.payload,
    )


def materialize(
    plan: LatticeSurgeryPlan,
    *,
    device: Device,
):
    """Materialize the exact solved plan as P2; never plan or project again."""

    if not isinstance(plan, LatticeSurgeryPlan):
        raise TypeError(
            "lattice-surgery materialization requires a solved plan")
    if not isinstance(device, Device):
        raise TypeError(
            "lattice-surgery materialization requires a typed device")
    if device.name != plan.device_name:
        raise ValueError(
            f"plan targets device @{plan.device_name}, not @{device.name}")
    compiler = _bound_device_compiler(device, plan.provider_key)
    actual_digest = compiler.architecture_digest(device)
    if actual_digest != plan.architecture_digest:
        raise _mapping_module().MappingVerificationError(
            "lattice-surgery plan architecture digest does not match device")
    selected_pipeline = compiler.pipeline
    if (_pipeline_digest(selected_pipeline)
            != plan.materialization_pipeline_digest):
        raise _mapping_module().MappingVerificationError(
            "lattice-surgery plan materialization-pipeline digest does not "
            "match the provider recipe")
    pipeline_verdict = compiler.accepts_pipeline(selected_pipeline)
    if type(pipeline_verdict) is not bool:
        raise TypeError(f"lattice-surgery compiler {compiler.key!r} "
                        "accepts_pipeline() must return bool")
    if not pipeline_verdict:
        raise ValueError(
            f"lattice-surgery compiler {compiler.key!r} does not implement "
            "the plan's P2N materialization pipeline")
    network_context = _network_context(plan, device, compiler)
    materialize_options = {"pipeline": selected_pipeline}
    if network_context is not None:
        materialize_options["context"] = network_context
    result = compiler.materialize_p2(plan, device, **materialize_options)
    from ...compiler import Build

    if not isinstance(result, Build):
        raise TypeError(
            f"lattice-surgery compiler {compiler.key!r} returned "
            f"{type(result).__name__}, expected cudaq.logical.Build")
    from cudaq.logical.stages import (
        P2,
        PATCH_GRAPH,
        PROTOCOL_NETWORK,
        QEC_SPEC,
    )

    required_facets = (QEC_SPEC, PROTOCOL_NETWORK, PATCH_GRAPH)
    root = result.definitions.get(result.root.symbol)
    if (result.profile != "p2n" or result.stage is not P2 or root is None or
            root.kind != "fabric.protocol" or
            any(facet not in result.facets for facet in required_facets) or
            result.schedule is not None or not result.verify()):
        raise ValueError(
            f"lattice-surgery compiler {compiler.key!r} returned an "
            "invalid P2N protocol build")
    if result.pipeline != selected_pipeline:
        raise ValueError(
            f"lattice-surgery compiler {compiler.key!r} returned a build "
            "with different pipeline provenance")
    plan_obligation = f"plan-digest={plan.digest}"
    if not any(evidence.producer == compiler.key and plan_obligation in
               evidence.obligations and evidence.result == "pass"
               for evidence in result.evidence):
        raise ValueError(
            f"lattice-surgery compiler {compiler.key!r} omitted exact-plan "
            "evidence")
    from ...compiler import CompilationContext

    transaction = CompilationContext.replay(result)
    if any(
            operation.name.startswith("phys.")
            for operation in transaction.walk()):
        raise ValueError(
            f"lattice-surgery compiler {compiler.key!r} placed physical "
            "operations in its P2 materialization")
    if network_context is not None:
        source = network_context.source
        if result.placement != source.placement:
            raise ValueError(
                "network P2 materialization did not retain the exact P1 "
                "placement witness")
        if result.qec_selection != network_context.selection:
            raise ValueError(
                "network P2 materialization did not retain the selected "
                "QECLowering witness")
        if result.evidence[:len(source.evidence)] != source.evidence:
            raise ValueError(
                "network P2 materialization did not retain the P1 evidence prefix"
            )
        if not set(source.source_modules) <= set(result.source_modules):
            raise ValueError(
                "network P2 materialization omitted a P1 linked source module")
        root_operation = transaction.find_symbol(result.root.symbol,
                                                 "fabric.protocol")
        if (root_operation is None or
                "generated_by" not in root_operation.attributes):
            raise ValueError(
                "network P2 protocol omitted its selected QECLowering provenance"
            )
        lowering_symbol = _attribute_text(
            root_operation.attributes["generated_by"])
        lowering_operation = transaction.find_symbol(lowering_symbol,
                                                     "qlx.qec_lowering")
        if lowering_operation is None:
            raise ValueError(
                "network P2 protocol generated_by does not resolve to a "
                "QECLowering manifest")
        metadata = (root_operation.attributes["metadata"]
                    if "metadata" in root_operation.attributes else None)
        if (metadata is None or "input_p1" not in metadata or
                "qec_selection_sha256" not in metadata or
                _attribute_text(metadata["input_p1"]) != source.root.symbol or
                _attribute_text(metadata["qec_selection_sha256"])
                != network_context.selection_digest):
            raise ValueError(
                "network P2 protocol omitted its exact P1/selection commitment")
        if ("manifest_name" not in lowering_operation.attributes or
                "manifest_sha256" not in lowering_operation.attributes or
                _attribute_text(lowering_operation.attributes["manifest_name"])
                != network_context.lowering.name or _attribute_text(
                    lowering_operation.attributes["manifest_sha256"])
                != network_context.lowering.manifest_sha256 or _attribute_text(
                    lowering_operation.attributes["compiler_plugin"])
                != network_context.lowering.plugin or _attribute_text(
                    lowering_operation.attributes["compiler_version"])
                != network_context.lowering.version or _attribute_text(
                    lowering_operation.attributes["compiler_symbol"])
                != compiler.name):
            raise ValueError(
                "network P2 QECLowering manifest does not identify the solved "
                "compiler")
    compiler.verify_p2_materialization(plan, device, result)
    return result


def apply(
    plan: LatticeSurgeryPlan,
    *,
    device: Device,
):
    """Compatibility spelling for exact-plan materialization."""

    return materialize(plan, device=device)


__all__ = [
    "AncillaPolicy",
    "Compatibility",
    "CompatibilityDiagnostic",
    "EpochCandidate",
    "EpochInfeasible",
    "EpochPlanningContext",
    "LatticeSurgeryCompiler",
    "LatticeSurgeryEpoch",
    "LatticeSurgeryOperation",
    "LatticeSurgeryPlan",
    "LatticeSurgeryProblem",
    "LatticeSurgeryProgram",
    "EncodedYInjection",
    "PauliTerm",
    "ProductMeasurement",
    "ProgramInstruction",
    "ProgramInstructionKind",
    "ProviderSolution",
    "QECNetworkAction",
    "QECNetworkArtifact",
    "QECNetworkEpoch",
    "QECNetworkPlan",
    "QECNetworkProjection",
    "QECNetworkRegion",
    "QECNetworkRegionBuilder",
    "QECNetworkRegionPlan",
    "QECNetworkRequest",
    "QECNetworkValue",
    "SurgeryPrimitives",
    "TemporalResource",
    "TemporalResourceClaim",
    "TemporalResourceKind",
    "YBasisPolicy",
    "apply",
    "connected_ancilla",
    "local_basis_change",
    "materialize",
    "mpp_compiler",
    "network_projection",
    "pack_epochs",
    "problem",
    "request",
    "solve",
    "surface_primitives",
    "validate_network_plan",
]

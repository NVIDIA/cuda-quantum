# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Inspectable, replayable P0-to-P1 logical placement.

The semantic seam is intentionally small:

``PlacementProblem -> PlacementPlan -> verified P1 Build``.

The built-in strategy is the canonical deterministic first-fit converter.  The
problem and plan are ordinary immutable JSON artifacts; MLIR mutation and P1
verification remain compiler-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cudaq.logical.architecture.constraints import (
    AllowSpaces,
    Colocate,
    DistributedPlacement,
    LocalPlacement,
    PlacementBinding,
    Prefer,
    RequireCapability,
    TopologicalPlacement,
    TrajectoryPlacement,
)
from cudaq.logical.architecture.logical import LogicalValueRef
from cudaq.logical.architecture.logical import (
    CapabilityKey,
    LogicalValueGroup,
    Space,
)

PROBLEM_SCHEMA = "qlx.placement-problem/v1"
PLAN_SCHEMA = "qlx.placement-plan/v1"
_UNSET = object()


def _mapping_module():
    # Import lazily so ``import cudaq.logical`` can initialize its compiler package before
    # publishing this research surface.
    from ..compiler import mapping

    return mapping


def _frozen(value):
    return _mapping_module().freeze_json(value)


def _thawed(value):
    return _mapping_module().thaw_json(value)


def _digest(value) -> str:
    return _mapping_module().artifact_digest(value)


def _raw_object(value, *, what: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{what} must be a JSON object")
    return value


def _raw_array(value, *, what: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{what} must be a JSON array")
    return value


def _raw_string(value, *, what: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{what} must be a JSON string")
    return value


def _raw_integer(value, *, what: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{what} must be a JSON integer")
    return value


def _require_exact_keys(value, expected, *, what: str) -> None:
    value = _raw_object(value, what=what)
    expected = frozenset(expected)
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(f"{what} has unsupported fields; "
                         f"missing={sorted(expected - actual)!r}, "
                         f"unexpected={sorted(actual - expected)!r}")


def _validate_reference_artifact(value, *, what: str) -> None:
    _require_exact_keys(
        value,
        ("program", "group", "path", "allocation"),
        what=what,
    )
    _raw_string(value["program"], what=f"{what}.program")
    _raw_string(value["group"], what=f"{what}.group")
    for index, item in enumerate(_raw_array(value["path"],
                                            what=f"{what}.path")):
        _raw_integer(item, what=f"{what}.path[{index}]")
    _raw_integer(value["allocation"], what=f"{what}.allocation")


def _validate_constraint_artifact(value, *, index: int) -> None:
    what = f"placement problem constraint[{index}]"
    value = _raw_object(value, what=what)
    kind = _raw_string(value.get("kind"), what=f"{what}.kind")
    fields = {
        "allow_spaces": ("kind", "spaces"),
        "require_capability": ("kind", "capability"),
        "colocate": ("kind", "values"),
        "local": ("kind", "value", "space", "slot", "witness"),
        "prefer": ("kind", "space", "role", "for"),
        "distributed": (
            "kind",
            "value",
            "spaces",
            "support_views",
            "ownership_witness",
            "link_obligations",
        ),
        "trajectory": (
            "kind",
            "value",
            "segments",
            "transition_events",
            "continuity_witness",
        ),
        "topological_record": (
            "kind",
            "value",
            "space",
            "record",
            "frontier",
            "support_witness",
            "observable_witness",
        ),
    }
    try:
        expected = fields[kind]
    except KeyError as exc:
        raise ValueError(f"{what} has unsupported kind {kind!r}") from exc
    _require_exact_keys(value, expected, what=what)

    if kind in {"allow_spaces", "distributed"}:
        for ordinal, item in enumerate(
                _raw_array(value["spaces"], what=f"{what}.spaces")):
            _raw_string(item, what=f"{what}.spaces[{ordinal}]")
    if kind == "require_capability":
        _raw_string(value["capability"], what=f"{what}.capability")
    if kind == "colocate":
        for ordinal, item in enumerate(
                _raw_array(value["values"], what=f"{what}.values")):
            _validate_reference_artifact(
                item,
                what=f"{what}.values[{ordinal}]",
            )
    if kind in {"local", "distributed", "trajectory", "topological_record"}:
        _validate_reference_artifact(value["value"], what=f"{what}.value")
    if kind in {"local", "topological_record"}:
        _raw_string(value["space"], what=f"{what}.space")
    if kind == "local" and value["slot"] is not None:
        _raw_integer(value["slot"], what=f"{what}.slot")
    if kind == "local" and value["witness"] is not None:
        _raw_string(value["witness"], what=f"{what}.witness")
    if kind == "distributed":
        for field in ("support_views", "link_obligations"):
            for ordinal, item in enumerate(
                    _raw_array(value[field], what=f"{what}.{field}")):
                _raw_string(
                    item,
                    what=f"{what}.{field}[{ordinal}]",
                )
        _raw_string(
            value["ownership_witness"],
            what=f"{what}.ownership_witness",
        )
    if kind == "trajectory":
        for ordinal, item in enumerate(
                _raw_array(value["segments"], what=f"{what}.segments")):
            _raw_string(item, what=f"{what}.segments[{ordinal}]")
        for ordinal, item in enumerate(
                _raw_array(value["transition_events"],
                           what=f"{what}.transition_events")):
            _raw_string(
                item,
                what=f"{what}.transition_events[{ordinal}]",
            )
        _raw_string(
            value["continuity_witness"],
            what=f"{what}.continuity_witness",
        )
    if kind == "topological_record":
        for field in (
                "record",
                "support_witness",
                "observable_witness",
        ):
            _raw_string(value[field], what=f"{what}.{field}")
        for ordinal, item in enumerate(
                _raw_array(value["frontier"], what=f"{what}.frontier")):
            _raw_string(item, what=f"{what}.frontier[{ordinal}]")
    if kind == "prefer":
        if value["space"] is not None:
            _raw_string(value["space"], what=f"{what}.space")
        if value["role"] is not None:
            _raw_string(value["role"], what=f"{what}.role")
        if value["for"] is None:
            return
        target = _raw_object(value["for"], what=f"{what}.for")
        target_kind = _raw_string(
            target.get("kind"),
            what=f"{what}.for.kind",
        )
        target_fields = {
            "value": ("kind", "value"),
            "group": ("kind", "program", "name", "count", "allocation"),
            "values": ("kind", "values"),
            "literal": ("kind", "value"),
        }
        try:
            _require_exact_keys(
                target,
                target_fields[target_kind],
                what=f"{what}.for",
            )
        except KeyError as exc:
            raise ValueError(
                f"{what}.for has unsupported kind {target_kind!r}") from exc
        if target_kind == "value":
            _validate_reference_artifact(
                target["value"],
                what=f"{what}.for.value",
            )
        elif target_kind == "group":
            _raw_string(target["program"], what=f"{what}.for.program")
            _raw_string(target["name"], what=f"{what}.for.name")
            _raw_integer(target["count"], what=f"{what}.for.count")
            _raw_integer(
                target["allocation"],
                what=f"{what}.for.allocation",
            )
        elif target_kind == "values":
            for ordinal, item in enumerate(
                    _raw_array(target["values"], what=f"{what}.for.values")):
                _validate_reference_artifact(
                    item,
                    what=f"{what}.for.values[{ordinal}]",
                )
        elif type(target["value"]) not in (str, bool, int, float):
            raise TypeError(f"{what}.for.value must be a JSON scalar")


def _validate_problem_artifact(value) -> None:
    what = "placement problem"
    _require_exact_keys(
        value,
        (
            "schema",
            "program",
            "program_digest",
            "machine",
            "machine_digest",
            "data",
            "constraints",
            "objective",
            "digest",
        ),
        what=what,
    )
    for key in (
            "schema",
            "program",
            "program_digest",
            "machine",
            "machine_digest",
            "objective",
            "digest",
    ):
        _raw_string(value[key], what=f"{what}.{key}")

    data = _raw_object(value["data"], what=f"{what}.data")
    _require_exact_keys(
        data,
        ("machine_contract", "value_groups", "objective_supplied"),
        what=f"{what}.data",
    )
    if type(data["objective_supplied"]) is not bool:
        raise TypeError(
            f"{what}.data.objective_supplied must be a JSON boolean")

    machine = _raw_object(
        data["machine_contract"],
        what=f"{what}.data.machine_contract",
    )
    _require_exact_keys(
        machine,
        ("name", "spaces", "streams", "channels"),
        what=f"{what}.data.machine_contract",
    )
    _raw_string(machine["name"], what=f"{what}.data.machine_contract.name")
    for index, item in enumerate(
            _raw_array(machine["spaces"],
                       what=f"{what}.data.machine_contract.spaces")):
        item_what = f"{what}.data.machine_contract.spaces[{index}]"
        _require_exact_keys(
            item,
            ("name", "capacity", "capabilities", "tags"),
            what=item_what,
        )
        _raw_string(item["name"], what=f"{item_what}.name")
        if item["capacity"] is not None:
            _raw_integer(item["capacity"], what=f"{item_what}.capacity")
        for field in ("capabilities", "tags"):
            for ordinal, entry in enumerate(
                    _raw_array(item[field], what=f"{item_what}.{field}")):
                _raw_string(
                    entry,
                    what=f"{item_what}.{field}[{ordinal}]",
                )
    for index, item in enumerate(
            _raw_array(machine["streams"],
                       what=f"{what}.data.machine_contract.streams")):
        item_what = f"{what}.data.machine_contract.streams[{index}]"
        _require_exact_keys(
            item,
            ("name", "produces", "buffer_size"),
            what=item_what,
        )
        _raw_string(item["name"], what=f"{item_what}.name")
        _raw_string(item["produces"], what=f"{item_what}.produces")
        _raw_integer(item["buffer_size"], what=f"{item_what}.buffer_size")
    for index, item in enumerate(
            _raw_array(machine["channels"],
                       what=f"{what}.data.machine_contract.channels")):
        item_what = f"{what}.data.machine_contract.channels[{index}]"
        _require_exact_keys(
            item,
            (
                "name",
                "source",
                "destination",
                "capabilities",
                "direction",
                "capacity",
            ),
            what=item_what,
        )
        for field in ("name", "source", "destination", "direction"):
            _raw_string(item[field], what=f"{item_what}.{field}")
        if item["capacity"] is not None:
            _raw_integer(item["capacity"], what=f"{item_what}.capacity")
        for ordinal, capability in enumerate(
                _raw_array(item["capabilities"],
                           what=f"{item_what}.capabilities")):
            _raw_string(
                capability,
                what=f"{item_what}.capabilities[{ordinal}]",
            )

    for index, item in enumerate(
            _raw_array(data["value_groups"], what=f"{what}.data.value_groups")):
        item_what = f"{what}.data.value_groups[{index}]"
        _require_exact_keys(
            item,
            ("allocation", "name", "count"),
            what=item_what,
        )
        _raw_integer(item["allocation"], what=f"{item_what}.allocation")
        _raw_string(item["name"], what=f"{item_what}.name")
        _raw_integer(item["count"], what=f"{item_what}.count")

    for index, item in enumerate(
            _raw_array(value["constraints"], what=f"{what}.constraints")):
        _validate_constraint_artifact(item, index=index)


def _validate_plan_artifact(value) -> None:
    what = "placement plan"
    _require_exact_keys(
        value,
        (
            "schema",
            "problem_digest",
            "strategy",
            "bindings",
            "metrics",
            "evidence",
            "digest",
        ),
        what=what,
    )
    for key in ("schema", "problem_digest", "strategy", "digest"):
        _raw_string(value[key], what=f"{what}.{key}")
    _raw_object(value["metrics"], what=f"{what}.metrics")
    _raw_object(value["evidence"], what=f"{what}.evidence")
    for index, item in enumerate(
            _raw_array(value["bindings"], what=f"{what}.bindings")):
        item_what = f"{what}.bindings[{index}]"
        _require_exact_keys(
            item,
            (
                "placement",
                "space",
                "slot",
                "source_allocation",
                "source_group",
                "source_path",
                "binding_kind",
                "binding_data",
            ),
            what=item_what,
        )
        for field in ("placement", "space", "binding_kind"):
            _raw_string(item[field], what=f"{item_what}.{field}")
        _raw_integer(item["slot"], what=f"{item_what}.slot")
        if item["source_allocation"] is not None:
            _raw_integer(
                item["source_allocation"],
                what=f"{item_what}.source_allocation",
            )
        if item["source_group"] is not None:
            _raw_string(
                item["source_group"],
                what=f"{item_what}.source_group",
            )
        for ordinal, path_item in enumerate(
                _raw_array(item["source_path"],
                           what=f"{item_what}.source_path")):
            _raw_integer(
                path_item,
                what=f"{item_what}.source_path[{ordinal}]",
            )
        for ordinal, pair in enumerate(
                _raw_array(item["binding_data"],
                           what=f"{item_what}.binding_data")):
            pair = _raw_array(
                pair,
                what=f"{item_what}.binding_data[{ordinal}]",
            )
            if len(pair) != 2:
                raise ValueError(
                    f"{item_what}.binding_data[{ordinal}] must have two items")
            _raw_string(
                pair[0],
                what=f"{item_what}.binding_data[{ordinal}][0]",
            )


def _value_groups(program) -> tuple[dict[str, Any], ...]:
    return tuple({
        "allocation": int(group.allocation),
        "name": str(group.name),
        "count": int(group.count),
    } for group in program.values)


def _program_payload(program) -> Mapping[str, Any]:
    return {
        "root": program.root.symbol,
        "profile": program.profile,
        "mlir": program.to_mlir(),
        "values": _value_groups(program),
    }


def _program_digest(program) -> str:
    return _digest(_program_payload(program))


def _qualified_name(value) -> str:
    for attribute in ("name", "key", "symbol"):
        item = getattr(value, attribute, None)
        if isinstance(item, str):
            return item
    return str(value)


def _direction(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _machine_contract(machine) -> Mapping[str, Any]:
    """Return exactly the logical facts consumed by the P0-to-P1 converter."""

    return {
        "name":
            machine.name,
        "spaces":
            tuple({
                "name":
                    space.name,
                "capacity":
                    space.capacity,
                "capabilities":
                    tuple(_qualified_name(item) for item in space.capabilities),
                "tags":
                    tuple(str(item) for item in space.tags),
            } for space in machine.spaces),
        "streams":
            tuple({
                "name": stream.name,
                "produces": _qualified_name(stream.produces),
                "buffer_size": stream.buffer_size,
            } for stream in machine.streams),
        "channels":
            tuple({
                "name":
                    channel.name,
                "source":
                    channel.source.name,
                "destination":
                    channel.destination.name,
                "capabilities":
                    tuple(
                        _qualified_name(item) for item in channel.capabilities),
                "direction":
                    _direction(channel.direction),
                "capacity":
                    channel.capacity,
            } for channel in machine.channels),
    }


def _reference_record(reference: LogicalValueRef) -> Mapping[str, Any]:
    return {
        "program": reference.program,
        "group": reference.group,
        "path": tuple(reference.path),
        "allocation": reference.allocation,
    }


def _canonical_value_group(value, program, *, what: str):
    required = ("program", "name", "count", "allocation")
    if any(not hasattr(value, field) for field in required):
        raise TypeError(
            f"{what} must be a logical value group from the supplied P0 program"
        )
    target = (
        value.program,
        value.name,
        value.count,
        value.allocation,
    )
    if (type(target[0]) is not str or type(target[1]) is not str or
            type(target[2]) is not int or type(target[3]) is not int):
        raise TypeError(f"{what} has invalid logical value group metadata")
    if target[0] != program.root.symbol:
        raise _mapping_module().MappingVerificationError(
            f"{what} references another P0 program")
    matches = tuple(group for group in program.values if (
        group.program,
        group.name,
        group.count,
        group.allocation,
    ) == target)
    if len(matches) != 1:
        raise _mapping_module().MappingVerificationError(
            f"{what} is absent or ambiguous in the supplied P0 program")
    return matches[0]


def _canonical_reference(reference, program, *, what: str) -> LogicalValueRef:
    if not isinstance(reference, LogicalValueRef):
        raise TypeError(f"{what} must be a logical value reference")
    if reference.program != program.root.symbol:
        raise _mapping_module().MappingVerificationError(
            f"{what} references another P0 program")
    if (type(reference.group) is not str or
            type(reference.allocation) is not int or
            type(reference.path) is not tuple or len(reference.path) != 1 or
            type(reference.path[0]) is not int):
        raise _mapping_module().MappingVerificationError(
            f"{what} is not a top-level logical value in the supplied P0 program"
        )
    matches = tuple(
        group for group in program.values
        if (group.program == reference.program and group.name == reference.group
            and group.allocation == reference.allocation))
    if len(matches) != 1:
        raise _mapping_module().MappingVerificationError(
            f"{what} names a missing or ambiguous logical value group")
    group = matches[0]
    index = reference.path[0]
    if index < 0 or index >= group.count:
        raise _mapping_module().MappingVerificationError(
            f"{what} has a path outside its logical value group")
    canonical = group[index]
    if canonical != reference:
        raise _mapping_module().MappingVerificationError(
            f"{what} does not match the supplied P0 value schema")
    return canonical


def _canonical_preference_target(value, program):
    if value is None:
        return None
    if isinstance(value, LogicalValueRef):
        return _canonical_reference(
            value,
            program,
            what="placement preference target",
        )
    if all(
            hasattr(value, field)
            for field in ("program", "name", "count", "allocation")):
        return _canonical_value_group(
            value,
            program,
            what="placement preference target",
        )
    if isinstance(value, (tuple, list)) and all(
            isinstance(item, LogicalValueRef) for item in value):
        return tuple(
            _canonical_reference(
                item,
                program,
                what=f"placement preference target value {index}",
            ) for index, item in enumerate(value))
    if type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "cudaq.logical.prefer for_= must be a logical value, logical value group, "
        "sequence of logical values, or JSON scalar")


def _preference_target(value) -> Any:
    if value is None:
        return None
    if isinstance(value, LogicalValueRef):
        return {"kind": "value", "value": _reference_record(value)}
    if hasattr(value, "program") and hasattr(value, "name") and hasattr(
            value, "count"):
        return {
            "kind": "group",
            "program": str(value.program),
            "name": str(value.name),
            "count": int(value.count),
            "allocation": int(value.allocation),
        }
    if isinstance(value, (tuple, list)) and all(
            isinstance(item, LogicalValueRef) for item in value):
        return {
            "kind": "values",
            "values": tuple(_reference_record(item) for item in value),
        }
    if isinstance(value, (str, bool, int, float)):
        return {"kind": "literal", "value": value}
    raise TypeError(
        "cudaq.logical.prefer for_= must be a logical value, logical value group, "
        "sequence of logical values, or JSON scalar")


def _canonical_space(value, machine, *, what: str) -> Space:
    if not isinstance(value, Space):
        raise TypeError(f"{what} must be a logical machine Space")
    matches = tuple(
        space for space in machine.spaces if space.name == value.name)
    if len(matches) != 1:
        raise _mapping_module().MappingVerificationError(
            f"{what} names a missing or ambiguous logical machine space")
    canonical = matches[0]
    if canonical != value:
        raise _mapping_module().MappingVerificationError(
            f"{what} does not match the supplied logical machine contract")
    return canonical


def _canonical_capability(value, machine, *, what: str) -> CapabilityKey:
    if not isinstance(value, CapabilityKey):
        raise TypeError(f"{what} must be a logical machine CapabilityKey")
    if type(value.key) is not str or not value.key:
        raise TypeError(f"{what} key must be a nonempty string")
    matches = tuple(capability for space in machine.spaces
                    for capability in space.capabilities
                    if _qualified_name(capability) == _qualified_name(value))
    if any(candidate != value for candidate in matches):
        raise _mapping_module().MappingVerificationError(
            f"{what} conflicts with the supplied logical machine contract")
    # Capability keys are open-vocabulary predicates, not machine-owned
    # resources. An absent key is a valid hard requirement whose solve is
    # infeasible; a present key reuses the machine's equal canonical value.
    return matches[0] if matches else CapabilityKey(value.key)


def _normalize_constraint(constraint, program, machine):
    if isinstance(constraint, AllowSpaces):
        return AllowSpaces(
            tuple(
                _canonical_space(
                    space,
                    machine,
                    what=f"allowed placement space {index}",
                ) for index, space in enumerate(constraint.spaces)))
    if isinstance(constraint, RequireCapability):
        return RequireCapability(
            _canonical_capability(
                constraint.capability,
                machine,
                what="required placement capability",
            ))
    if isinstance(constraint, Colocate):
        return Colocate(
            tuple(
                _canonical_reference(
                    value,
                    program,
                    what=f"colocation value {index}",
                ) for index, value in enumerate(constraint.values)))
    if isinstance(constraint, LocalPlacement):
        return LocalPlacement(
            _canonical_reference(
                constraint.value,
                program,
                what="local placement value",
            ),
            _canonical_space(
                constraint.space,
                machine,
                what="local placement space",
            ),
            constraint.slot,
            constraint.witness,
        )
    if isinstance(constraint, Prefer):
        return Prefer(
            space=(None if constraint.space is None else _canonical_space(
                constraint.space,
                machine,
                what="preferred placement space",
            )),
            role=constraint.role,
            for_=_canonical_preference_target(constraint.for_, program),
        )
    if isinstance(constraint, DistributedPlacement):
        return DistributedPlacement(
            _canonical_reference(
                constraint.value,
                program,
                what="distributed placement value",
            ),
            tuple(
                _canonical_space(
                    space,
                    machine,
                    what=f"distributed placement space {index}",
                ) for index, space in enumerate(constraint.spaces)),
            constraint.support_views,
            constraint.ownership_witness,
            constraint.link_obligations,
        )
    if isinstance(constraint, TrajectoryPlacement):
        return TrajectoryPlacement(
            _canonical_reference(
                constraint.value,
                program,
                what="trajectory placement value",
            ),
            tuple(
                _canonical_space(
                    space,
                    machine,
                    what=f"trajectory placement segment {index}",
                ) for index, space in enumerate(constraint.segments)),
            constraint.transition_events,
            constraint.continuity_witness,
        )
    if isinstance(constraint, TopologicalPlacement):
        return TopologicalPlacement(
            _canonical_reference(
                constraint.value,
                program,
                what="topological placement value",
            ),
            _canonical_space(
                constraint.space,
                machine,
                what="topological placement space",
            ),
            constraint.record,
            constraint.frontier,
            constraint.support_witness,
            constraint.observable_witness,
        )
    return constraint


def _constraint_record(constraint) -> Mapping[str, Any]:
    if isinstance(constraint, AllowSpaces):
        return {
            "kind": "allow_spaces",
            "spaces": tuple(space.name for space in constraint.spaces),
        }
    if isinstance(constraint, RequireCapability):
        return {
            "kind": "require_capability",
            "capability": _qualified_name(constraint.capability),
        }
    if isinstance(constraint, Colocate):
        return {
            "kind":
                "colocate",
            "values":
                tuple(
                    _reference_record(reference)
                    for reference in constraint.values),
        }
    if isinstance(constraint, LocalPlacement):
        return {
            "kind": "local",
            "value": _reference_record(constraint.value),
            "space": constraint.space.name,
            "slot": constraint.slot,
            "witness": constraint.witness,
        }
    if isinstance(constraint, Prefer):
        return {
            "kind":
                "prefer",
            "space":
                None if constraint.space is None else constraint.space.name,
            "role":
                constraint.role,
            "for":
                _preference_target(constraint.for_),
        }
    if isinstance(constraint, DistributedPlacement):
        return {
            "kind": "distributed",
            "value": _reference_record(constraint.value),
            "spaces": tuple(space.name for space in constraint.spaces),
            "support_views": constraint.support_views,
            "ownership_witness": constraint.ownership_witness,
            "link_obligations": constraint.link_obligations,
        }
    if isinstance(constraint, TrajectoryPlacement):
        return {
            "kind": "trajectory",
            "value": _reference_record(constraint.value),
            "segments": tuple(space.name for space in constraint.segments),
            "transition_events": constraint.transition_events,
            "continuity_witness": constraint.continuity_witness,
        }
    if isinstance(constraint, TopologicalPlacement):
        return {
            "kind": "topological_record",
            "value": _reference_record(constraint.value),
            "space": constraint.space.name,
            "record": constraint.record,
            "frontier": constraint.frontier,
            "support_witness": constraint.support_witness,
            "observable_witness": constraint.observable_witness,
        }
    raise TypeError(
        "placement constraints must be typed qlx placement descriptors, got "
        f"{type(constraint).__name__}")


def _binding_record(binding: PlacementBinding) -> Mapping[str, Any]:
    return {
        "placement":
            binding.placement,
        "space":
            binding.space,
        "slot":
            binding.slot,
        "source_allocation":
            binding.source_allocation,
        "source_group":
            binding.source_group,
        "source_path":
            binding.source_path,
        "binding_kind":
            binding.binding_kind,
        "binding_data":
            tuple((str(key), _thawed(value))
                  for key, value in binding.binding_data),
    }


def _binding_from_record(record: Mapping[str, Any]) -> PlacementBinding:
    return PlacementBinding(
        placement=str(record["placement"]),
        space=str(record["space"]),
        slot=int(record["slot"]),
        source_allocation=(None if record.get("source_allocation") is None else
                           int(record["source_allocation"])),
        source_group=(None if record.get("source_group") is None else str(
            record["source_group"])),
        source_path=tuple(int(item) for item in record.get("source_path", ())),
        binding_kind=str(record.get("binding_kind", "local")),
        binding_data=tuple((str(item[0]), _restore_binding_value(item[1]))
                           for item in record.get("binding_data", ())),
    )


def _restore_binding_value(value):
    if isinstance(value, list):
        return tuple(_restore_binding_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((str(key), _restore_binding_value(item))
                     for key, item in sorted(value.items()))
    return value


def _problem_digest_payload(
    *,
    schema: str,
    program: str,
    program_digest: str,
    machine: str,
    machine_digest: str,
    data: Mapping[str, Any],
    constraints,
    objective: str,
) -> Mapping[str, Any]:
    return {
        "schema": schema,
        "program": program,
        "program_digest": program_digest,
        "machine": machine,
        "machine_digest": machine_digest,
        "data": data,
        "constraints": constraints,
        "objective": objective,
    }


@dataclass(frozen=True, slots=True)
class PlacementProblem:
    """Immutable, JSON-replayable logical placement problem."""

    schema: str
    program: str
    program_digest: str
    machine: str
    machine_digest: str
    data: Mapping[str, Any]
    constraints: tuple[Mapping[str, Any], ...]
    objective: str
    digest: str
    _program: Any = field(default=None, repr=False, compare=False)
    _device: Any = field(default=None, repr=False, compare=False)
    _constraint_objects: tuple[Any, ...] = field(default=(),
                                                 repr=False,
                                                 compare=False)
    _objective_value: Any = field(default=_UNSET, repr=False, compare=False)
    _experiment: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != PROBLEM_SCHEMA:
            raise ValueError(
                f"unsupported placement problem schema {self.schema!r}")
        object.__setattr__(self, "data", _frozen(self.data))
        object.__setattr__(
            self,
            "constraints",
            tuple(_frozen(item) for item in self.constraints),
        )
        machine_contract = self.data["machine_contract"]
        embedded_machine_digest = _digest(machine_contract)
        if embedded_machine_digest != self.machine_digest:
            raise _mapping_module().MappingVerificationError(
                "placement problem machine_digest does not certify its "
                "embedded logical machine contract")
        if machine_contract["name"] != self.machine:
            raise _mapping_module().MappingVerificationError(
                "placement problem machine name does not match its embedded "
                "logical machine contract")
        payload = _problem_digest_payload(
            schema=self.schema,
            program=self.program,
            program_digest=self.program_digest,
            machine=self.machine,
            machine_digest=self.machine_digest,
            data=self.data,
            constraints=self.constraints,
            objective=self.objective,
        )
        _mapping_module().verify_artifact_digest(payload,
                                                 self.digest,
                                                 artifact="placement problem")

    @classmethod
    def _create(
        cls,
        *,
        program,
        device,
        machine,
        constraints,
        objective,
        experiment,
    ) -> "PlacementProblem":
        machine_contract = _machine_contract(machine)
        machine_digest = _digest(machine_contract)
        program_digest = _program_digest(program)
        constraints = tuple(
            _normalize_constraint(item, program, machine)
            for item in constraints)
        records = tuple(_constraint_record(item) for item in constraints)
        data = {
            "machine_contract": machine_contract,
            "value_groups": _value_groups(program),
            "objective_supplied": objective is not None,
        }
        normalized_objective = str(objective or "first_fit")
        payload = _problem_digest_payload(
            schema=PROBLEM_SCHEMA,
            program=program.root.symbol,
            program_digest=program_digest,
            machine=machine.name,
            machine_digest=machine_digest,
            data=data,
            constraints=records,
            objective=normalized_objective,
        )
        return cls(
            schema=PROBLEM_SCHEMA,
            program=program.root.symbol,
            program_digest=program_digest,
            machine=machine.name,
            machine_digest=machine_digest,
            data=data,
            constraints=records,
            objective=normalized_objective,
            digest=_digest(payload),
            _program=program,
            _device=device,
            _constraint_objects=constraints,
            _objective_value=objective,
            _experiment=experiment,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _problem_digest_payload(
            schema=self.schema,
            program=self.program,
            program_digest=self.program_digest,
            machine=self.machine,
            machine_digest=self.machine_digest,
            data=self.data,
            constraints=self.constraints,
            objective=self.objective,
        )
        return {**_thawed(payload), "digest": self.digest}

    def to_json(self) -> str:
        return _mapping_module().canonical_json(self.to_dict())

    def save(self, path: str | Path) -> None:
        _mapping_module().save_artifact(path, self.to_dict())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        program=None,
        device=None,
    ) -> "PlacementProblem":
        value = _mapping_module().load_artifact(path)
        _validate_problem_artifact(value)
        bound_program = None if program is None else _as_p0(program)
        loaded = cls(
            schema=value["schema"],
            program=value["program"],
            program_digest=value["program_digest"],
            machine=value["machine"],
            machine_digest=value["machine_digest"],
            data=value["data"],
            constraints=tuple(value["constraints"]),
            objective=value["objective"],
            digest=value["digest"],
            _program=bound_program,
            _device=device,
            _objective_value=(value["objective"] if value["data"].get(
                "objective_supplied", False) else None),
        )
        if bound_program is not None:
            _require_program(loaded, bound_program)
        if device is not None:
            _device, machine = _require_device(loaded, device)
            if bound_program is not None:
                object.__setattr__(
                    loaded,
                    "_constraint_objects",
                    _constraints_from_records(
                        loaded,
                        bound_program,
                        machine,
                    ),
                )
        return loaded

    def spaces(self) -> tuple[str, ...]:
        return tuple(
            str(item["name"])
            for item in self.data["machine_contract"]["spaces"])

    def capacity(self, space: str) -> int | None:
        for item in self.data["machine_contract"]["spaces"]:
            if item["name"] == space:
                value = item["capacity"]
                return None if value is None else int(value)
        raise KeyError(space)


def _plan_digest_payload(
    *,
    schema: str,
    problem_digest: str,
    strategy: str,
    bindings,
    metrics,
    evidence,
) -> Mapping[str, Any]:
    return {
        "schema": schema,
        "problem_digest": problem_digest,
        "strategy": strategy,
        "bindings": bindings,
        "metrics": metrics,
        "evidence": evidence,
    }


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Immutable placement decisions tied to one exact problem digest."""

    schema: str
    problem_digest: str
    strategy: str
    bindings: tuple[PlacementBinding, ...]
    metrics: Mapping[str, Any]
    evidence: Mapping[str, Any]
    digest: str
    _build: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise ValueError(
                f"unsupported placement plan schema {self.schema!r}")
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if any(not isinstance(item, PlacementBinding)
               for item in self.bindings):
            raise TypeError(
                "PlacementPlan bindings must be PlacementBinding values")
        object.__setattr__(self, "metrics", _frozen(self.metrics))
        object.__setattr__(self, "evidence", _frozen(self.evidence))
        payload = _plan_digest_payload(
            schema=self.schema,
            problem_digest=self.problem_digest,
            strategy=self.strategy,
            bindings=tuple(_binding_record(item) for item in self.bindings),
            metrics=self.metrics,
            evidence=self.evidence,
        )
        _mapping_module().verify_artifact_digest(payload,
                                                 self.digest,
                                                 artifact="placement plan")

    @classmethod
    def _create(
        cls,
        *,
        problem_digest: str,
        strategy: str,
        bindings,
        metrics,
        evidence,
        build=None,
    ) -> "PlacementPlan":
        bindings = tuple(bindings)
        payload = _plan_digest_payload(
            schema=PLAN_SCHEMA,
            problem_digest=problem_digest,
            strategy=strategy,
            bindings=tuple(_binding_record(item) for item in bindings),
            metrics=metrics,
            evidence=evidence,
        )
        return cls(
            schema=PLAN_SCHEMA,
            problem_digest=problem_digest,
            strategy=strategy,
            bindings=bindings,
            metrics=metrics,
            evidence=evidence,
            digest=_digest(payload),
            _build=build,
        )

    @property
    def assignments(self) -> Mapping[str, Any]:
        values = {}
        for binding in self.bindings:
            if binding.source_allocation is not None:
                path = ".".join(str(item) for item in binding.source_path)
                key = f"allocation:{binding.source_allocation}"
                if path:
                    key += f"[{path}]"
            elif binding.source_group is not None:
                path = ".".join(str(item) for item in binding.source_path)
                key = f"group:{binding.source_group}"
                if path:
                    key += f"[{path}]"
            else:
                key = f"placement:{binding.placement}"
            values[key] = {
                "space": binding.space,
                "slot": binding.slot,
                "kind": binding.binding_kind,
            }
        return _frozen(values)

    def to_dict(self) -> dict[str, Any]:
        payload = _plan_digest_payload(
            schema=self.schema,
            problem_digest=self.problem_digest,
            strategy=self.strategy,
            bindings=tuple(_binding_record(item) for item in self.bindings),
            metrics=self.metrics,
            evidence=self.evidence,
        )
        return {**_thawed(payload), "digest": self.digest}

    def to_json(self) -> str:
        return _mapping_module().canonical_json(self.to_dict())

    def save(self, path: str | Path) -> None:
        _mapping_module().save_artifact(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "PlacementPlan":
        value = _mapping_module().load_artifact(path)
        _validate_plan_artifact(value)
        return cls(
            schema=value["schema"],
            problem_digest=value["problem_digest"],
            strategy=value["strategy"],
            bindings=tuple(
                _binding_from_record(item) for item in value["bindings"]),
            metrics=value["metrics"],
            evidence=value["evidence"],
            digest=value["digest"],
        )


def _as_p0(program):
    from ..compiler.build import Build
    from ..compiler.compile import compile
    from ..compiler.pipeline import pipelines

    if not isinstance(program, Build):
        program = compile(program, pipeline=pipelines.logical())
    if program.profile != "p0":
        raise ValueError(
            "placement research API requires a P0 Build or portable definition")
    return program


def _require_program(problem: PlacementProblem, program):
    program = _as_p0(program)
    actual = _program_digest(program)
    if actual != problem.program_digest:
        raise _mapping_module().MappingVerificationError(
            "placement problem was built for another P0 program: "
            f"expected {problem.program_digest!r}, computed {actual!r}")
    if program.root.symbol != problem.program:
        raise _mapping_module().MappingVerificationError(
            f"placement problem expects @{problem.program}, "
            f"got @{program.root.symbol}")
    if tuple(problem.data["value_groups"]) != _value_groups(program):
        raise _mapping_module().MappingVerificationError(
            "placement problem value_groups do not match the supplied P0 "
            "program")
    return program


def _require_device(problem: PlacementProblem, device):
    from ..compiler.place import _as_machine

    machine = _as_machine(device)
    live_contract = _machine_contract(machine)
    actual = _digest(live_contract)
    if actual != problem.machine_digest:
        raise _mapping_module().MappingVerificationError(
            "placement problem was built for another logical machine: "
            f"expected {problem.machine_digest!r}, computed {actual!r}")
    if machine.name != problem.machine:
        raise _mapping_module().MappingVerificationError(
            f"placement problem expects machine @{problem.machine}, "
            f"got @{machine.name}")
    if live_contract != problem.data["machine_contract"]:
        raise _mapping_module().MappingVerificationError(
            "placement problem embedded logical machine contract does not "
            "match the supplied device")
    return device, machine


def _reference_from_record(record: Mapping[str, Any],
                           program) -> LogicalValueRef:
    reference = LogicalValueRef(
        program=str(record["program"]),
        group=str(record["group"]),
        path=tuple(int(item) for item in record.get("path", ())),
        allocation=(None if record.get("allocation") is None else int(
            record["allocation"])),
    )
    return _canonical_reference(
        reference,
        program,
        what="serialized placement constraint reference",
    )


def _preference_target_from_record(record, program):
    if record is None:
        return None
    kind = record["kind"]
    if kind == "value":
        return _reference_from_record(record["value"], program)
    if kind == "values":
        return tuple(
            _reference_from_record(item, program) for item in record["values"])
    if kind == "literal":
        return record["value"]
    if kind != "group":
        raise _mapping_module().MappingVerificationError(
            f"serialized preference has unsupported target kind {kind!r}")

    return _canonical_value_group(
        LogicalValueGroup(
            record["program"],
            record["name"],
            record["count"],
            record["allocation"],
        ),
        program,
        what="serialized placement preference group",
    )


def _constraints_from_records(problem, program, machine):
    spaces = {space.name: space for space in machine.spaces}
    capabilities = {
        _qualified_name(capability): capability for space in machine.spaces
        for capability in space.capabilities
    }

    def space(name):
        try:
            return spaces[str(name)]
        except KeyError as exc:
            raise _mapping_module().MappingVerificationError(
                f"serialized placement constraint names missing space {name!r}"
            ) from exc

    restored = []
    for record in problem.constraints:
        kind = record["kind"]
        if kind == "allow_spaces":
            restored.append(
                AllowSpaces(tuple(space(item) for item in record["spaces"])))
        elif kind == "require_capability":
            name = str(record["capability"])
            capability = capabilities.get(name, CapabilityKey(name))
            restored.append(RequireCapability(capability))
        elif kind == "colocate":
            restored.append(
                Colocate(
                    tuple(
                        _reference_from_record(item, program)
                        for item in record["values"])))
        elif kind == "local":
            restored.append(
                LocalPlacement(
                    _reference_from_record(record["value"], program),
                    space(record["space"]),
                    (None
                     if record.get("slot") is None else int(record["slot"])),
                    record.get("witness"),
                ))
        elif kind == "prefer":
            restored.append(
                Prefer(
                    space=(None if record.get("space") is None else space(
                        record["space"])),
                    role=record.get("role"),
                    for_=_preference_target_from_record(
                        record.get("for"),
                        program,
                    ),
                ))
        elif kind == "distributed":
            restored.append(
                DistributedPlacement(
                    _reference_from_record(record["value"], program),
                    tuple(space(item) for item in record["spaces"]),
                    tuple(str(item) for item in record["support_views"]),
                    str(record["ownership_witness"]),
                    tuple(
                        str(item)
                        for item in record.get("link_obligations", ())),
                ))
        elif kind == "trajectory":
            restored.append(
                TrajectoryPlacement(
                    _reference_from_record(record["value"], program),
                    tuple(space(item) for item in record["segments"]),
                    tuple(str(item) for item in record["transition_events"]),
                    str(record["continuity_witness"]),
                ))
        elif kind == "topological_record":
            restored.append(
                TopologicalPlacement(
                    _reference_from_record(record["value"], program),
                    space(record["space"]),
                    str(record["record"]),
                    tuple(str(item) for item in record["frontier"]),
                    str(record["support_witness"]),
                    str(record["observable_witness"]),
                ))
        else:
            raise _mapping_module().MappingVerificationError(
                "detached placement problem cannot reconstruct opaque "
                f"constraint {record.get('type', '<unknown>')!r}")
    return tuple(restored)


def _context(
    problem: PlacementProblem,
    *,
    program=None,
    device=None,
):
    selected_program = problem._program if program is None else program
    if selected_program is None:
        raise _mapping_module().MappingVerificationError(
            "detached placement problem requires program=")
    selected_device = problem._device if device is None else device
    if selected_device is None:
        raise _mapping_module().MappingVerificationError(
            "detached placement problem requires device=")
    program = _require_program(problem, selected_program)
    device, machine = _require_device(problem, selected_device)
    constraints = (problem._constraint_objects if problem._constraint_objects
                   else _constraints_from_records(problem, program, machine))
    objective = (problem._objective_value if problem._objective_value
                 is not _UNSET else problem.objective)
    return program, device, constraints, objective


def problem(
    program,
    *,
    device,
    placement=(),
    constraints=None,
    objective=None,
    experiment=None,
) -> PlacementProblem:
    """Analyze one portable program against a logical device contract."""

    program = _as_p0(program)
    if constraints is not None:
        if placement:
            raise TypeError("specify placement= or constraints=, not both")
        placement = constraints
    if callable(placement):
        placement = placement(program.values)
    normalized = tuple(placement or ())
    from ..compiler.place import _as_machine

    machine = _as_machine(device)
    return PlacementProblem._create(
        program=program,
        device=device,
        machine=machine,
        constraints=normalized,
        objective=objective,
        experiment=experiment,
    )


def _first_fit(problem: PlacementProblem, *, program=None, device=None):
    program, device, constraints, objective = _context(problem,
                                                       program=program,
                                                       device=device)
    from ..compiler.place import _place_build

    build = _place_build(
        program,
        device=device,
        placement=constraints,
        objective=objective,
        experiment=problem._experiment,
    )
    spaces = tuple(sorted({item.space for item in build.placement.bindings}))
    return PlacementPlan._create(
        problem_digest=problem.digest,
        strategy="first_fit",
        bindings=build.placement.bindings,
        metrics={
            "bindings": len(build.placement.bindings),
            "spaces_used": len(spaces),
        },
        evidence={
            "solver": "qlx-first-fit",
            "version": "1",
            "tie_break": "declaration_order",
            "objective": problem.objective,
        },
        build=build,
    )


def solve(
    problem: PlacementProblem,
    *,
    strategy: str = "first_fit",
    program=None,
    device=None,
) -> PlacementPlan:
    """Solve one placement problem with deterministic ``first_fit``.

    Callable strategies are intentionally not accepted in this first research
    slice: :func:`apply` can independently reconstruct and verify only the
    compiler-owned first-fit decision.
    """

    if not isinstance(problem, PlacementProblem):
        raise TypeError("cudaq.logical.architecture.placement.solve requires a "
                        "PlacementProblem")
    if callable(strategy):
        raise TypeError(
            "callable placement strategies are not executable in this slice; "
            "use strategy='first_fit'")
    if strategy != "first_fit":
        raise ValueError(
            f"unknown placement strategy {strategy!r}; available: ('first_fit',)"
        )
    return _first_fit(problem, program=program, device=device)


def apply(
    program,
    problem: PlacementProblem,
    plan: PlacementPlan,
    *,
    device=None,
    experiment=None,
):
    """Verify and materialize one exact placement plan as canonical P1."""

    if not isinstance(problem, PlacementProblem):
        raise TypeError("cudaq.logical.architecture.placement.apply requires a "
                        "PlacementProblem")
    if not isinstance(plan, PlacementPlan):
        raise TypeError("cudaq.logical.architecture.placement.apply requires a "
                        "PlacementPlan")
    _mapping_module().verify_plan_for_problem(
        problem_digest=problem.digest,
        plan_problem_digest=plan.problem_digest,
    )
    program, device, constraints, objective = _context(problem,
                                                       program=program,
                                                       device=device)
    if plan.strategy != "first_fit":
        raise _mapping_module().MappingVerificationError(
            "this implementation can materialize only first_fit placement plans"
        )
    if plan._build is not None and experiment is None:
        candidate = plan._build
    else:
        from ..compiler.place import _place_build

        candidate = _place_build(
            program,
            device=device,
            placement=constraints,
            objective=objective,
            experiment=problem._experiment
            if experiment is None else experiment,
        )
    if tuple(candidate.placement.bindings) != tuple(plan.bindings):
        raise _mapping_module().MappingVerificationError(
            "placement plan decisions do not match compiler-owned first-fit "
            "materialization")
    if (candidate.placement.machine != problem.machine or
            candidate.placement.input_p0 != problem.program or
            candidate.placement.objective != problem.objective):
        raise _mapping_module().MappingVerificationError(
            "materialized P1 placement witness does not match its problem")
    return candidate


__all__ = [
    "PLAN_SCHEMA",
    "PROBLEM_SCHEMA",
    "PlacementPlan",
    "PlacementProblem",
    "apply",
    "problem",
    "solve",
]

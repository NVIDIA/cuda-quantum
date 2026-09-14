# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import ClassVar, Mapping

from .types import Tier, _EstimateResult


@dataclass(frozen=True, slots=True)
class LogicalProfile(_EstimateResult):
    annotation_tier: ClassVar[Tier] = Tier.LOGICAL

    actions: Mapping[str, int]
    instruments: Mapping[str, int]
    idle_sites: int
    discards: int
    logical_qubits_peak: int = 0
    action_depth_upper_bound: int = 0
    synthesis_demand: Mapping[str, int] | None = None
    resource_requests: Mapping[str, int] | None = None
    resource_consumptions: Mapping[str, int] | None = None
    build_root: str = ""
    build_sha256: str = ""

    @property
    def total_operations(self) -> int:
        return (sum(self.actions.values()) + sum(self.instruments.values()) +
                self.idle_sites + sum(
                    (self.resource_requests or {}).values()) + sum(
                        (self.resource_consumptions or {}).values()))


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _symbol_attribute(operation, name: str) -> str:
    attr = operation.attributes[name]
    value = getattr(attr, "value", None)
    if value is not None:
        return str(value)
    return str(attr).lstrip("@")


def _children(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child.operation


def _symbol(operation):
    try:
        return _symbol_attribute(operation, "sym_name")
    except KeyError:
        return None


def logical_counts(build) -> LogicalProfile:
    if build.profile != "p0":
        raise ValueError("logical_counts requires a P0 Build")
    actions: Counter[str] = Counter()
    instruments: Counter[str] = Counter()
    synthesis: Counter[str] = Counter()
    resource_requests: Counter[str] = Counter()
    resource_consumptions: Counter[str] = Counter()
    idle_sites = discards = depth = 0
    # Analysis is a compiler consumer, not a UI inspection. Read from a fresh
    # clone of the sealed authority so mutation of the public inspection view
    # cannot change an estimate.
    module = build._fresh_module()
    symbols = {
        _symbol(operation.operation): operation.operation
        for operation in module.body.operations
        if _symbol(operation.operation) is not None
    }
    root = symbols.get(build.root.symbol)
    if root is None:
        raise ValueError(
            f"build root @{build.root.symbol} is absent from its module")

    def visit(operation, multiplier=1, call_stack=()):
        nonlocal idle_sites, discards, depth
        name = operation.name
        if name == "cflow.repeat":
            count = int(operation.attributes["count"])
            for child in _children(operation):
                visit(child, multiplier * count, call_stack)
            return
        if name == "qlx.call":
            callee = _symbol_attribute(operation, "callee")
            target = symbols.get(callee)
            if target is None:
                raise ValueError(f"unresolved logical call @{callee}")
            if callee in call_stack:
                raise ValueError(
                    f"recursive logical call graph through @{callee}")
            for child in _children(target):
                visit(child, multiplier, (*call_stack, callee))
            return
        if name == "qlx.apply":
            action_attr = str(operation.attributes["action"])
            action = ("qlx_standard_" +
                      action_attr[len("#qlx.action<"):-1].strip('"')
                      if action_attr.startswith("#qlx.action<") else
                      _symbol_attribute(operation, "action"))
            actions[action] += multiplier
            depth += multiplier
            if any(token in action for token in ("_t", "ccz", "ccx")):
                synthesis[action] += multiplier
        elif name == "qlx.instrument":
            instrument_attr = str(operation.attributes["instrument"])
            instrument = ("qlx_standard_" +
                          instrument_attr[len("#qlx.instrument<"):-1].strip('"')
                          if instrument_attr.startswith("#qlx.instrument<") else
                          _symbol_attribute(operation, "instrument"))
            instruments[instrument] += multiplier
            depth += multiplier
        elif name == "qlx.prepare":
            state = str(operation.attributes["state"]).strip('"')
            instruments[f"qlx_standard_prepare_{state}"] += multiplier
            depth += multiplier
        elif name == "qlx.measure":
            basis = str(operation.attributes["basis"])
            basis = basis[len("#qlx.pauli<"):-1].strip('"').lower()
            instruments[f"qlx_standard_measure_{basis}"] += multiplier
            depth += multiplier
        elif name == "qlx.idle":
            idle_sites += multiplier
            depth += multiplier
        elif name == "qlx.discard":
            discards += multiplier
        elif name == "qlx.resource_request":
            resource_requests[_symbol_attribute(operation,
                                                "kind")] += multiplier
        elif name == "qlx.consume_resource":
            resource_type = str(operation.operands[0].type)
            prefix = '!qlx.logical_resource<"'
            resource_kind = (resource_type[len(prefix):-2]
                             if resource_type.startswith(prefix) else
                             resource_type)
            resource_consumptions[resource_kind] += multiplier
            action_attr = str(operation.attributes["action"])
            action = ("qlx_standard_" +
                      action_attr[len("#qlx.action<"):-1].strip('"')
                      if action_attr.startswith("#qlx.action<") else
                      _symbol_attribute(operation, "action"))
            synthesis[action] += multiplier
            depth += multiplier
        for child in _children(operation):
            visit(child, multiplier, call_stack)

    for child in _children(root):
        visit(child, call_stack=(build.root.symbol,))

    arg_qubits = 0
    if root.regions and root.regions[0].blocks:
        arg_qubits = sum(
            str(argument.type) == "!qlx.logical_qubit"
            for argument in root.regions[0].blocks[0].arguments)

    def live_profile(block, live, stack):
        peak = live
        for view in block.operations:
            operation = view.operation
            name = operation.name
            if name == "qlx.prepare":
                live += 1
            elif name == "qlx.measure":
                live -= sum(
                    str(value.type) == "!qlx.logical_qubit"
                    for value in operation.operands)
            elif name == "qlx.discard":
                live -= sum(
                    str(value.type) == "!qlx.logical_qubit"
                    for value in operation.operands)
            elif name == "cflow.repeat":
                count = int(operation.attributes["count"])
                if count and operation.regions and operation.regions[0].blocks:
                    final, nested_peak = live_profile(
                        operation.regions[0].blocks[0], live, stack)
                    if final != live:
                        raise ValueError(
                            "cflow.repeat changes live logical ownership across "
                            "an iteration")
                    peak = max(peak, nested_peak)
            elif name == "cflow.if":
                branch_finals = []
                for region in operation.regions:
                    if not region.blocks:
                        branch_finals.append(live)
                        continue
                    final, branch_peak = live_profile(region.blocks[0], live,
                                                      stack)
                    branch_finals.append(final)
                    peak = max(peak, branch_peak)
                if branch_finals and any(
                        final != branch_finals[0] for final in branch_finals):
                    raise ValueError(
                        "cflow.if branches disagree on live logical ownership")
                if branch_finals:
                    live = branch_finals[0]
            elif name == "qlx.call":
                callee = _symbol_attribute(operation, "callee")
                if callee in stack:
                    raise ValueError(
                        f"recursive logical call graph through @{callee}")
                target = symbols.get(callee)
                if target is None or not target.regions or not target.regions[
                        0].blocks:
                    raise ValueError(f"unresolved logical call @{callee}")
                live, call_peak = live_profile(target.regions[0].blocks[0],
                                               live, (*stack, callee))
                peak = max(peak, call_peak)
            if live < 0:
                raise ValueError("logical ownership accounting became negative")
            peak = max(peak, live)
        return live, peak

    _, logical_peak = live_profile(root.regions[0].blocks[0], arg_qubits,
                                   (build.root.symbol,))
    return LogicalProfile(
        actions=dict(actions),
        instruments=dict(instruments),
        idle_sites=idle_sites,
        discards=discards,
        logical_qubits_peak=logical_peak,
        action_depth_upper_bound=depth,
        synthesis_demand=dict(synthesis),
        resource_requests=dict(resource_requests),
        resource_consumptions=dict(resource_consumptions),
        build_root=build.root.symbol,
        build_sha256=build.content_sha256,
    )


LogicalEstimate = LogicalProfile

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Insert and remove explicit idle events in physical builds.

Idle is a first-class, composable facet of a physical build rather than
something the compiler bakes in or a backend emitter invents:

* :func:`add_idle` inserts a typed ``phys.delay`` event on every carrier that
  sits idle through a scheduled tick (the gate layers from
  :func:`cudaq.logical.ticks`).
* :func:`remove_idle` strips the idle events -- the inverse of :func:`add_idle`.

Each is a ``Build -> Build`` transform, so the workflow composes and a target
only ever serializes explicit idle events.
"""

from __future__ import annotations

import re

import cudaq.mlir.ir as mlir_ir

from .build import Build, EvidenceRecord
from .context import CompilationContext
from .schedule import schedule as _schedule
from .ticks import ticks as _ticks

_CARRIER = re.compile(r"@([A-Za-z0-9_.]+)")


def _carrier(value) -> str | None:
    """The carrier symbol of a ``!phys.state<@carrier>`` SSA value, or None."""
    text = str(value.type)
    if "phys.state" not in text:
        return None
    match = _CARRIER.search(text)
    return match.group(1) if match else None


def _carrier_bindings(module) -> dict[str, str]:
    """Map allocation-local resource symbols to scheduler carrier keys."""
    bindings = {}
    for view in module.body.operations:
        operation = view.operation
        if operation.name != "phys.resource":
            continue
        symbol = _text(operation.attributes["sym_name"])
        resource_class = _text(operation.attributes["resource_class"])
        index = int(operation.attributes["index"])
        bindings[symbol] = f"{resource_class}[{index}]"
    return bindings


def _binding(value, bindings) -> str | None:
    symbol = _carrier(value)
    return bindings.get(symbol, symbol)


def _graph_block(transaction, root_symbol):
    root = transaction.find_symbol(root_symbol)
    graph_symbol = root_symbol
    if root is not None and "graph" in root.attributes:
        graph_symbol = _text(root.attributes["graph"])
    graph = transaction.find_symbol(graph_symbol, "phys.graph")
    if graph is None:
        raise ValueError("idle transforms require a P3 phys.graph build")
    return graph, graph.regions[0].blocks[0]


def _drop_schedule(module, root_symbol):
    """A build whose SSA thread changes carries a stale phys.schedule; drop it."""
    for view in list(module.body.operations):
        operation = view.operation
        if operation.name == "phys.schedule":
            operation.erase()


def _rebuild(transaction, source, module, *, extra_evidence=()):
    facets = tuple(f for f in source.facets if f != "physical_schedule")
    return Build(
        context=transaction.context,
        module=module,
        root=source.root,
        profile=source.profile,
        facets=facets,
        pipeline=source.pipeline,
        evidence=(*source.evidence, *extra_evidence),
        placement=source.placement,
        qec_selection=source.qec_selection,
        experiment=source.experiment,
        source_modules=source.source_modules,
    )


def add_idle(build: Build, *, duration_ns: float = 1.0) -> Build:
    """Insert a typed ``phys.delay`` on every carrier idle through a tick.

    Uses :func:`cudaq.logical.ticks` (the scheduled gate layers) to find, for
    each tick,
    the in-play carriers that are not gated, and splices one ``phys.delay`` event
    per idle tick into that carrier's state thread. Returns a new build; the old
    ``phys.schedule`` (if any) is dropped because the thread changed -- re-run
    :func:`cudaq.logical.schedule` afterwards.
    """
    if not isinstance(build, Build) or build.profile != "p3":
        raise TypeError("add_idle requires a P3 Build")

    physical_schedule = _schedule(build)
    ticks = _ticks(physical_schedule)
    tick_of_event = {
        event: tick.index for tick in ticks for event in tick.events
    }
    idle_ticks: dict[str, set[int]] = {}
    for tick in ticks:
        for carrier in tick.idle:
            idle_ticks.setdefault(carrier, set()).add(tick.index)

    transaction = CompilationContext.replay(build)
    module = transaction.module
    _graph, block = _graph_block(transaction, build.root.symbol)
    carrier_bindings = _carrier_bindings(module)

    f64 = None
    with transaction.location:
        f64 = mlir_ir.F64Type.get(context=transaction.context)

    counter = 0
    last_tick: dict[str, int] = {}
    for view in list(block.operations):
        operation = view.operation
        if operation.name == "phys.acquire":
            for result in operation.results:
                carrier = _binding(result, carrier_bindings)
                if carrier is not None:
                    last_tick.pop(carrier, None)
        event = operation.attributes.get("event_id")
        this_tick = (tick_of_event.get(_text(event))
                     if event is not None else None)
        operands = list(operation.operands)
        for position, operand in enumerate(operands):
            carrier = _binding(operand, carrier_bindings)
            if carrier is None:
                continue
            prior = last_tick.get(carrier)
            if prior is not None and this_tick is not None:
                gap = sorted(t for t in idle_ticks.get(carrier, ())
                             if prior < t < this_tick)
                if gap:
                    threaded = operand
                    for _ in gap:
                        with transaction.location:
                            delay = mlir_ir.Operation.create(
                                "phys.delay",
                                operands=[threaded],
                                results=[threaded.type],
                                attributes={
                                    "duration_ns":
                                        mlir_ir.FloatAttr.get(
                                            f64, float(duration_ns)),
                                    "event_id":
                                        mlir_ir.StringAttr.get(
                                            f"idle{counter}",
                                            context=transaction.context),
                                },
                            )
                        counter += 1
                        mlir_ir.InsertionPoint(operation).insert(delay)
                        threaded = delay.results[0]
                    operation.operands[position] = threaded
        # after this op, its results define each carrier's current value + tick
        for result in operation.results:
            carrier = _binding(result, carrier_bindings)
            if carrier is not None and this_tick is not None:
                last_tick[carrier] = this_tick
        if operation.name == "phys.release":
            for operand in operation.operands:
                carrier = _binding(operand, carrier_bindings)
                if carrier is not None:
                    last_tick.pop(carrier, None)

    if not module.operation.verify():
        raise ValueError("add_idle produced an invalid module")
    _drop_schedule(module, build.root.symbol)
    return _rebuild(
        transaction,
        build,
        module,
        extra_evidence=(EvidenceRecord(kind="idle_events",
                                       producer="cudaq-logical-python@0.3",
                                       result=f"added:{counter}"),),
    )


def remove_idle(build: Build) -> Build:
    """Strip the idle ``phys.delay`` events -- the inverse of :func:`add_idle`."""
    if not isinstance(build, Build) or build.profile != "p3":
        raise TypeError("remove_idle requires a P3 Build")
    transaction = CompilationContext.replay(build)
    module = transaction.module
    _graph, block = _graph_block(transaction, build.root.symbol)
    removed = 0
    for view in list(block.operations):
        operation = view.operation
        if operation.name != "phys.delay":
            continue
        event = operation.attributes.get("event_id")
        if event is None or not _text(event).startswith("idle"):
            continue
        for result, operand in zip(operation.results, operation.operands):
            result.replace_all_uses_with(operand)
        operation.erase()
        removed += 1
    if not module.operation.verify():
        raise ValueError("remove_idle produced an invalid module")
    _drop_schedule(module, build.root.symbol)
    return _rebuild(
        transaction,
        build,
        module,
        extra_evidence=(EvidenceRecord(kind="idle_events",
                                       producer="cudaq-logical-python@0.3",
                                       result=f"removed:{removed}"),),
    )


def _text(attribute):
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


__all__ = ["add_idle", "remove_idle"]

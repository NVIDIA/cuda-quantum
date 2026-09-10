# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import json

from .finalize import public_finalizer


def _text(attribute) -> str:
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _value(attribute):
    value = getattr(attribute, "value", None)
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(attribute, "type") and str(attribute).startswith("dense<"):
        return [int(item) for item in attribute]
    try:
        named = list(attribute)
    except TypeError:
        named = None
    if named is not None:
        if named and hasattr(named[0], "name") and hasattr(named[0], "attr"):
            return {str(item.name): _value(item.attr) for item in named}
        return [_value(item) for item in named]
    return str(attribute).strip('"')


def _symbol(operation):
    if "sym_name" not in operation.attributes:
        return None
    return _text(operation.attributes["sym_name"])


def _find_top_level(module, symbol, operation_name=None):
    for view in module.body.operations:
        operation = view.operation
        if _symbol(operation) != symbol:
            continue
        if operation_name is None or operation.name == operation_name:
            return operation
    return None


def _attributes(operation, *, omit=()):
    omitted = set(omit)
    return {
        str(name): _value(attribute)
        for name, attribute in operation.attributes.items()
        if str(name) not in omitted
    }


def _architecture(operation):
    block = operation.regions[0].blocks[0]
    resources = []
    topologies = []
    bindings = []
    for view in block.operations:
        child = view.operation
        data = {
            "name": _symbol(child),
            **_attributes(child, omit=("sym_name",))
        }
        if child.name == "phys.resource_class":
            resources.append(data)
        elif child.name == "phys.topology":
            topologies.append(data)
        elif child.name == "phys.qec_binding":
            bindings.append(data)
    return {
        "name":
            _symbol(operation),
        **_attributes(operation, omit=("sym_name",)),
        "resource_classes":
            resources,
        "topologies":
            topologies,
        "qec_bindings":
            bindings,
    }


def _actions(module):
    return [{
        "name": _symbol(operation),
        **_attributes(operation, omit=("sym_name",))
    }
            for view in module.body.operations
            if (operation := view.operation).name == "phys.action"]


def _instruments(module):
    return [{
        "name": _symbol(operation),
        **_attributes(operation, omit=("sym_name",))
    }
            for view in module.body.operations
            if (operation := view.operation).name == "phys.instrument"]


def _graph(operation):
    block = operation.regions[0].blocks[0]
    value_ids = {}
    next_value = 0
    resources = []
    events = []
    outputs = []

    for view in block.operations:
        child = view.operation
        if child.name == "phys.return":
            outputs = [value_ids[value] for value in child.operands]
            continue
        if child.name == "phys.resource":
            resources.append({
                "name": _symbol(child),
                **_attributes(child, omit=("sym_name",))
            })
            continue

        inputs = [value_ids[value] for value in child.operands]
        result_ids = []
        for result in child.results:
            result_id = f"v{next_value}"
            next_value += 1
            value_ids[result] = result_id
            result_ids.append(result_id)
        events.append({
            "op": child.name.removeprefix("phys."),
            "inputs": inputs,
            "outputs": result_ids,
            "result_types": [str(result.type) for result in child.results],
            "attributes": _attributes(child),
        })
    return {
        "name":
            _symbol(operation),
        **_attributes(operation, omit=("sym_name", "function_type")),
        "resources":
            resources,
        "events":
            events,
        "outputs":
            outputs,
    }


def _schedules(module, graph_symbol):
    schedules = []
    for view in module.body.operations:
        operation = view.operation
        if operation.name != "phys.schedule":
            continue
        if _text(operation.attributes["graph"]) != graph_symbol:
            continue
        schedules.append({
            "name": _symbol(operation),
            **_attributes(operation, omit=("sym_name",))
        })
    return schedules


def module_has_resource_kind(module, kind: str) -> bool:
    for view in module.body.operations:
        architecture = view.operation
        if architecture.name != "phys.machine":
            continue
        for child_view in architecture.regions[0].blocks[0].operations:
            child = child_view.operation
            if (child.name == "phys.resource_class" and
                    _text(child.attributes["kind"]) == kind):
                return True
    return False


def physical_manifest_text():

    def finalize(module, ctx, *, root_symbol, **_):
        graph = _find_top_level(module, root_symbol, "phys.graph")
        if graph is None:
            raise ValueError(
                f"P3 target cannot find physical graph @{root_symbol}")
        architecture_symbol = _text(graph.attributes["architecture"])
        architecture = _find_top_level(module, architecture_symbol,
                                       "phys.machine")
        if architecture is None:
            raise ValueError(
                f"physical graph @{root_symbol} references missing architecture "
                f"@{architecture_symbol}")
        artifact = {
            "schema": "qlx.physical-manifest/v1",
            "architecture": _architecture(architecture),
            "actions": _actions(module),
            "instruments": _instruments(module),
            "graph": _graph(graph),
            "schedules": _schedules(module, root_symbol),
        }
        return json.dumps(artifact, indent=2, sort_keys=True) + "\n"

    return public_finalizer(finalize, "physical-manifest")


__all__ = ["module_has_resource_kind", "physical_manifest_text"]

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from .finalize import public_finalizer

import math


def _text(attribute):
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _symbol(operation):
    return _text(operation.attributes["sym_name"]
                ) if "sym_name" in operation.attributes else None


def _resource(type_):
    text = str(type_)
    prefix = "!phys.state<@"
    return text[len(prefix):-1] if text.startswith(prefix) else None


def _walk(operation):
    """Yield an operation and all nested operations in lexical order."""

    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _float(value):
    return f"{float(value):.6f}"


def _turns(radians):
    """Convert QLX's canonical radian angles to Lanes full turns."""

    return float(radians) / math.tau


class _LanesEmitter:
    """Conservative PEG-to-bloqade-lanes v1 text projection.

    The common PEG carries state ownership while ``phys.action`` supplies the
    target semantic.  This emitter deliberately fails on event families for
    which the v1 lanes stack machine has no faithful projection.
    """

    def __init__(self, module, root_symbol):
        self.module = module
        self.root_symbol = root_symbol
        operations = (operation for view in module.body.operations
                      for operation in _walk(view.operation))
        self.symbols = {
            _symbol(operation): operation
            for operation in operations
            if _symbol(operation) is not None
        }
        try:
            self.graph = self.symbols[root_symbol]
        except KeyError as exc:
            raise ValueError(
                f"Lanes target cannot find physical graph @{root_symbol}"
            ) from exc
        if self.graph.name != "phys.graph":
            raise ValueError("Lanes target requires a rooted phys.graph")
        resources = [
            operation for operation in self.symbols.values()
            if operation.name == "phys.resource"
        ]
        self.resource_class_capabilities = {
            _symbol(operation):
                frozenset(
                    _text(value)
                    for value in operation.attributes.get("capabilities", ()))
            for operation in self.symbols.values()
            if operation.name == "phys.resource_class"
        }
        resource_classes = sorted({
            _text(operation.attributes["resource_class"])
            for operation in resources
        })
        if len(resource_classes) > 1:
            raise ValueError(
                "Lanes bytecode v1 requires one physical resource class "
                "(controller address domain); found "
                f"{resource_classes!r}")
        bindings = {}
        self.resource_aliases = {}
        self.resource_classes = {}
        self.resource_indices = {}
        for operation in resources:
            symbol = _symbol(operation)
            resource_class = _text(operation.attributes["resource_class"])
            declaration = self.symbols.get(resource_class)
            if declaration is None or declaration.name != "phys.resource_class":
                raise ValueError(
                    f"Lanes resource @{symbol} references unknown resource "
                    f"class @{resource_class}")
            index = int(operation.attributes["index"])
            resource_count = int(declaration.attributes["count"])
            if index < 0 or index >= resource_count:
                raise ValueError(
                    f"Lanes resource @{symbol} index {index} is outside "
                    f"resource class @{resource_class} count {resource_count}")
            binding = (
                resource_class,
                index,
            )
            physical_resource = bindings.setdefault(binding, symbol)
            self.resource_aliases[symbol] = physical_resource
            self.resource_classes[physical_resource] = resource_class
            self.resource_indices[physical_resource] = binding[1]
        zones = [
            operation for operation in self.symbols.values()
            if operation.name == "phys.topology" and
            _text(operation.attributes["kind"]) == "zone"
            # Loading reservoirs feed ``initial_fill`` and are not an
            # addressable compute zone in the Lanes v1 instruction set.
            and _text(operation.attributes["parameters"]["role"]) != "loading"
        ]
        self.zone_indices = {
            _symbol(operation): index for index, operation in enumerate(zones)
        }
        self.zone_roles = {}
        self.zone_symbol_roles = {}
        for operation in zones:
            parameters = operation.attributes["parameters"]
            if "role" not in parameters:
                raise ValueError(
                    f"Lanes zone @{_symbol(operation)} is missing its role")
            role = _text(parameters["role"])
            symbol = _symbol(operation)
            if role in self.zone_roles:
                raise ValueError(
                    f"Lanes v1 requires one zone per role; {role!r} is ambiguous"
                )
            self.zone_roles[role] = symbol
            self.zone_symbol_roles[symbol] = role
        self.zone_parameters = {
            _symbol(operation): operation.attributes["parameters"]
            for operation in zones
        }
        initial_zone = self.zone_roles.get("storage")
        if resources and initial_zone is None:
            raise ValueError(
                "Lanes v1 requires a declared STORAGE zone for acquired atoms")
        self.resource_zones = {
            resource: initial_zone for resource in self.resource_indices
        }
        # ``initial_fill`` is the only v1 operation that establishes a known
        # loaded state. A zero/plus preparation may therefore be elided only
        # for a resource that has not participated in any earlier event.
        self.fresh_resources = set(self.resource_indices)
        self.active_owners = {}
        self.current_states = {}
        self.lines = [".version 1"]

    def _zone_word(self, role):
        try:
            zone = self.zone_roles[role]
            index = self.zone_indices[zone]
        except KeyError as exc:
            raise ValueError(
                f"Lanes v1 requires a declared {role.upper()} zone") from exc
        return f"0x{index:08x}"

    def _global_r(self, rotation_angle, axis_angle):
        # Lanes pops the axis angle first, then the rotation angle, and its
        # controller ABI measures both in full turns.
        self.lines.extend((
            f"const_float {_float(_turns(rotation_angle))}",
            f"const_float {_float(_turns(axis_angle))}",
            "global_r",
        ))

    def _global_rz(self, rotation_angle):
        self.lines.extend((
            f"const_float {_float(_turns(rotation_angle))}",
            "global_rz",
        ))

    def _require_resource_capability(self, resources, capability):
        resources = tuple(resources)
        if not resources:
            raise ValueError(
                f"Lanes {capability} operation requires physical resources")
        missing = sorted({
            self.resource_classes[resource]
            for resource in resources
            if capability not in self.resource_class_capabilities.get(
                self.resource_classes[resource], ())
        })
        if missing:
            raise ValueError(
                f"Lanes controller-global pulse requires resource class(es) "
                f"{missing!r} to declare {capability!r}")

    def _zone_coordinates(self, resource, zone=None):
        zone = self.resource_zones.get(resource) if zone is None else zone
        if zone not in self.zone_indices:
            raise ValueError(
                f"Lanes cannot resolve resource @{resource} to a declared zone")
        zone_index = self.zone_indices[zone]
        parameters = self.zone_parameters.get(zone)
        if (parameters is None or "words" not in parameters or
                "sites_per_word" not in parameters):
            raise ValueError(
                f"Lanes zone @{zone} must declare words and sites_per_word")
        words = int(_text(parameters["words"]))
        sites_per_word = int(_text(parameters["sites_per_word"]))
        if words <= 0 or sites_per_word <= 0:
            raise ValueError(f"Lanes zone @{zone} dimensions must be positive")
        index = self.resource_indices[resource]
        capacity = words * sites_per_word
        if index >= capacity:
            raise ValueError(
                f"Lanes resource @{resource} index {index} exceeds zone "
                f"@{zone} capacity {capacity}")
        word, site = divmod(index, sites_per_word)
        return zone_index, word, site

    def _location_word(self, resource):
        zone_index, word, site = self._zone_coordinates(resource)
        return (zone_index << 56) | (word << 40) | (site << 24)

    def _lane_word(self, resource):
        zone_index, word, site = self._zone_coordinates(resource)
        return (zone_index << 53) | (word << 16) | site

    def _local_r(self, resources, rotation_angle, axis_angle):
        resources = tuple(resources)
        for resource in resources:
            self.lines.append(
                f"const_loc 0x{self._location_word(resource):016x}")
        self.lines.extend((
            f"const_float {_float(_turns(rotation_angle))}",
            f"const_float {_float(_turns(axis_angle))}",
            f"local_r {len(resources)}",
        ))

    def _local_rz(self, resources, rotation_angle):
        resources = tuple(resources)
        for resource in resources:
            self.lines.append(
                f"const_loc 0x{self._location_word(resource):016x}")
        self.lines.extend((
            f"const_float {_float(_turns(rotation_angle))}",
            f"local_rz {len(resources)}",
        ))

    def _hadamard(self, resources, *, broadcast=False):
        # Under the Lanes U3 convention:
        #   Rz(1/2 turn) = Z
        #   R(axis=1/4 turn, rotation=1/4 turn) = Ry(pi/2)
        # and applying them in that order gives Ry(pi/2) Z = H exactly.
        if broadcast:
            self._global_rz(math.pi)
            self._global_r(math.pi / 2, math.pi / 2)
        else:
            self._local_rz(resources, math.pi)
            self._local_r(resources, math.pi / 2, math.pi / 2)

    def _action(self, action):
        declaration = self.symbols.get(action)
        if declaration is None:
            raise ValueError(
                "Lanes target requires a linked phys.action declaration "
                f"for @{action}")
        if declaration.name != "phys.action":
            raise ValueError(
                f"Lanes target expected @{action} to resolve to phys.action, "
                f"not {declaration.name}")
        semantics = {
            str(named.name): _text(named.attr)
            for named in declaration.attributes.get("controller_bindings", ())
        }
        try:
            return (
                semantics["lanes"].lower(),
                "broadcast" in declaration.attributes,
            )
        except KeyError as exc:
            raise NotImplementedError(
                f"physical action @{action} has no Lanes target semantics"
            ) from exc

    def _instrument_semantic(self, instrument):
        declaration = self.symbols.get(instrument)
        if declaration is None:
            raise ValueError(
                "Lanes target requires a linked phys.instrument declaration "
                f"for @{instrument}")
        if declaration.name != "phys.instrument":
            raise ValueError(
                f"Lanes target expected @{instrument} to resolve to "
                f"phys.instrument, not {declaration.name}")
        semantics = {
            str(named.name): _text(named.attr)
            for named in declaration.attributes.get("controller_bindings", ())
        }
        try:
            return semantics["lanes"].lower()
        except KeyError as exc:
            raise NotImplementedError(
                f"physical instrument @{instrument} has no Lanes target semantics"
            ) from exc

    def _emit_action(self, semantic, resources=(), *, broadcast=False):
        resources = tuple(resources)
        if broadcast and semantic in {
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "x",
                "y",
                "z",
        }:
            self._require_resource_capability(resources, "global_pulses")
            self._require_exact_global_effect(resources, semantic.upper())
            resource_classes = {
                self.resource_classes[resource] for resource in resources
            }
            self._mark_used(
                resource
                for resource, resource_class in self.resource_classes.items()
                if resource_class in resource_classes)
        if semantic == "h":
            self._hadamard(resources, broadcast=broadcast)
        elif semantic == "s":
            self._global_rz(math.pi / 2) if broadcast else self._local_rz(
                resources, math.pi / 2)
        elif semantic == "sdg":
            self._global_rz(-math.pi / 2) if broadcast else self._local_rz(
                resources, -math.pi / 2)
        elif semantic == "t":
            self._global_rz(math.pi / 4) if broadcast else self._local_rz(
                resources, math.pi / 4)
        elif semantic == "tdg":
            self._global_rz(-math.pi / 4) if broadcast else self._local_rz(
                resources, -math.pi / 4)
        elif semantic == "x":
            if broadcast:
                self._global_r(math.pi, 0)
            else:
                self._local_r(resources, math.pi, 0)
        elif semantic == "y":
            if broadcast:
                self._global_r(math.pi, math.pi / 2)
            else:
                self._local_r(resources, math.pi, math.pi / 2)
        elif semantic == "z":
            self._global_rz(math.pi) if broadcast else self._local_rz(
                resources, math.pi)
        elif semantic == "cz":
            entangling = self.zone_roles.get("entangling")
            self._require_exact_zone_effect(resources, entangling, "CZ")
            self._mark_used(
                resource for resource, zone in self.resource_zones.items()
                if zone == entangling)
            self.lines.extend(
                (f"const_zone {self._zone_word('entangling')}", "cz"))
        elif semantic == "reset":
            raise NotImplementedError(
                "Lanes v1 has no state-preserving reset/reload instruction")
        else:
            raise NotImplementedError(
                f"Lanes v1 does not support physical semantic {semantic!r}")

    def _consume_initial_fill(self, resources, state):
        resources = tuple(resources)
        stale = [
            resource for resource in resources
            if resource not in self.fresh_resources
        ]
        if stale:
            raise NotImplementedError(
                "Lanes v1 can elide zero/plus preparation only immediately "
                "after initial_fill; later preparation requires reset/reload "
                f"support (state={state!r}, resources={stale!r})")
        self.fresh_resources.difference_update(resources)

    def _mark_used(self, resources):
        self.fresh_resources.difference_update(resources)

    def _require_live_states(self, state_inputs, operation):
        stale = []
        seen = set()
        for value, (resource, owner) in state_inputs:
            current_owner = self.active_owners.get(resource)
            current_state = self.current_states.get(resource)
            if (resource in seen or current_owner is None or
                    current_owner != owner or current_state is None or
                    current_state != value):
                stale.append(
                    (resource, owner, current_owner, value, current_state))
            seen.add(resource)
        if stale:
            raise ValueError(
                f"Lanes {operation} requires states owned by their current "
                "allocation lifetime and current SSA version; "
                f"stale states={stale!r}")

    def _propagate_state_results(self, operation, input_states, states):
        results = self._state_results(operation)
        if len(results) != len(input_states):
            raise ValueError(
                f"Lanes {operation.name} does not preserve physical-state "
                "ownership one-for-one")
        for result, state in zip(results, input_states):
            resource, _owner = state
            states[result] = state
            self.current_states[resource] = result

    def _require_exact_global_effect(self, resources, operation):
        selected = set(resources)
        resource_classes = {
            self.resource_classes[resource] for resource in selected
        }
        occupants = {
            resource for resource in self.active_owners
            if self.resource_classes[resource] in resource_classes
        }
        if selected != occupants:
            raise ValueError(
                f"Lanes v1 controller-global {operation} requires selected "
                f"resources to exhaust the live controller scope; "
                f"unselected resources={sorted(occupants - selected)!r}, "
                f"selected outside scope={sorted(selected - occupants)!r}")

    def _require_exact_zone_effect(self, resources, zone, operation):
        selected = set(resources)
        occupants = {
            resource for resource, current_zone in self.resource_zones.items()
            if current_zone == zone and resource in self.active_owners
        }
        if selected != occupants:
            unselected = sorted(occupants - selected)
            absent = sorted(selected - occupants)
            raise ValueError(
                f"Lanes v1 zone-wide {operation} requires selected resources "
                f"to exhaust occupied zone @{zone}; "
                f"unselected occupants={unselected!r}, "
                f"selected outside zone={absent!r}")

    def _require_entangling_pair(self, resources):
        resources = tuple(resources)
        if len(resources) != 2 or resources[0] == resources[1]:
            raise ValueError(
                "Lanes v1 CZ requires two distinct physical resources")
        zone = self.zone_roles.get("entangling")
        if zone is None:
            raise ValueError("Lanes v1 requires a declared ENTANGLING zone")
        misplaced = [
            resource for resource in resources
            if self.resource_zones.get(resource) != zone
        ]
        if misplaced:
            raise ValueError(
                "Lanes v1 CZ requires both resources to occupy the declared "
                f"ENTANGLING zone @{zone}; misplaced resources={misplaced!r}")

    @staticmethod
    def _state_results(operation):
        return [
            result for result in operation.results
            if _resource(result.type) is not None
        ]

    def _emit_block(self, block, states):
        yielded = ()
        operations = [view.operation for view in block.operations]
        cursor = 0
        while cursor < len(operations):
            operation = operations[cursor]
            name = operation.name
            state_inputs = [(value, states[value])
                            for value in operation.operands
                            if _resource(value.type) is not None]
            input_states = [state for _value, state in state_inputs]
            inputs = [resource for resource, _owner in input_states]
            if name != "phys.acquire" and state_inputs:
                self._require_live_states(state_inputs, name)
            if name == "phys.acquire":
                for result in self._state_results(operation):
                    alias = _resource(result.type)
                    resource = self.resource_aliases[alias]
                    if resource in self.active_owners:
                        raise ValueError(
                            f"Lanes resource alias @{alias} acquires already-live "
                            f"physical binding @{self.resource_classes[resource]}"
                            f"[{self.resource_indices[resource]}]")
                    # The acquire result itself is the lifetime token. Resource
                    # symbols may be reused across disjoint acquires in
                    # hand-authored PEGs, so the symbol alone cannot
                    # distinguish a stale predecessor from the current owner.
                    states[result] = (resource, result)
                    self.active_owners[resource] = result
                    self.current_states[resource] = result
            elif name == "phys.prepare":
                state = _text(operation.attributes["state"])
                if state in {"plus", "+", "zero", "0", "loaded"}:
                    self._consume_initial_fill(inputs, state)
                if state in {"plus", "+"}:
                    self._emit_action("h", inputs)
                elif state not in {"zero", "0", "loaded"}:
                    raise NotImplementedError(
                        f"Lanes v1 does not support preparation {state!r}")
                self._propagate_state_results(operation, input_states, states)
            elif name == "phys.reset":
                state = _text(operation.attributes["state"])
                if state == "plus":
                    self._emit_action("reset", inputs)
                    self._emit_action("h", inputs)
                elif state == "zero":
                    self._emit_action("reset", inputs)
                else:
                    raise NotImplementedError(
                        f"Lanes v1 does not support reset {state!r}")
                self._propagate_state_results(operation, input_states, states)
            elif name == "phys.move":
                self._mark_used(inputs)
                route_name = _text(operation.attributes["route"])
                route = self.symbols.get(route_name)
                if route is not None and "parameters" in route.attributes:
                    parameters = route.attributes["parameters"]
                    source = _text(
                        parameters["source"]) if "source" in parameters else "?"
                    destination = (_text(parameters["destination"])
                                   if "destination" in parameters else "?")
                    trajectory = (_text(operation.attributes["trajectory"])
                                  if "trajectory" in operation.attributes else
                                  "shuttle")
                    self.lines.append(
                        f"; {trajectory}: {source} -> {destination}")
                    if "source" in parameters:
                        misplaced = [
                            resource for resource in inputs
                            if self.resource_zones.get(resource) != source
                        ]
                        if misplaced:
                            raise ValueError(
                                f"Lanes move via @{route_name} requires its "
                                f"resources to occupy source zone @{source}; "
                                f"misplaced resources={misplaced!r}")
                    if "destination" in parameters:
                        for resource in inputs:
                            self._zone_coordinates(resource, destination)
                for resource in inputs:
                    self.lines.append(
                        f"const_lane 0x{self._lane_word(resource):016x}")
                self.lines.append(f"move {len(inputs)}")
                if route is not None and "parameters" in route.attributes:
                    parameters = route.attributes["parameters"]
                    if "destination" in parameters:
                        destination = _text(parameters["destination"])
                        for resource in inputs:
                            self.resource_zones[resource] = destination
                self._propagate_state_results(operation, input_states, states)
            elif name == "phys.apply":
                semantic, broadcast = self._action(
                    _text(operation.attributes["action"]))
                if semantic == "cz" and not broadcast:
                    # Lanes CZ is a zone pulse. Consecutive disjoint pair
                    # events therefore form one matching layer, not one pulse
                    # per pair. Movement boundaries prevent accidental fusion
                    # across independently isolated blockade pairs.
                    seen = set()
                    batch = []
                    while cursor < len(operations):
                        candidate = operations[cursor]
                        if candidate.name != "phys.apply":
                            break
                        candidate_semantic, candidate_broadcast = self._action(
                            _text(candidate.attributes["action"]))
                        if candidate_semantic != "cz" or candidate_broadcast:
                            break
                        candidate_state_inputs = [
                            (value, states[value])
                            for value in candidate.operands
                            if _resource(value.type) is not None
                        ]
                        candidate_states = [
                            state for _value, state in candidate_state_inputs
                        ]
                        self._require_live_states(candidate_state_inputs,
                                                  candidate.name)
                        candidate_inputs = [
                            resource for resource, _owner in candidate_states
                        ]
                        self._require_entangling_pair(candidate_inputs)
                        if seen.intersection(candidate_inputs):
                            break
                        seen.update(candidate_inputs)
                        self._mark_used(candidate_inputs)
                        batch.append((candidate, candidate_inputs))
                        self._propagate_state_results(candidate,
                                                      candidate_states, states)
                        cursor += 1
                    self._emit_action("cz", tuple(seen))
                    continue
                if semantic == "cz":
                    raise ValueError(
                        "Lanes v1 CZ must be a non-broadcast pair event")
                self._mark_used(inputs)
                self._emit_action(semantic, inputs, broadcast=broadcast)
                self._propagate_state_results(operation, input_states, states)
            elif name == "phys.measure":
                # Measurement is zone-wide in Lanes. Fold a consecutive group
                # of scalar PEG records into one readout-zone instruction while
                # retaining each PEG record/result in the state map.
                zone = self.resource_zones.get(inputs[0]) if inputs else None
                if (zone is None or
                        self.zone_symbol_roles.get(zone) != "readout"):
                    raise ValueError(
                        "Lanes v1 measurement requires the resource to occupy "
                        "a declared READOUT zone")
                restore_x = []
                seen = set()
                while cursor < len(operations):
                    candidate = operations[cursor]
                    if candidate.name != "phys.measure":
                        break
                    candidate_state_inputs = [
                        (value, states[value])
                        for value in candidate.operands
                        if _resource(value.type) is not None
                    ]
                    candidate_states = [
                        state for _value, state in candidate_state_inputs
                    ]
                    self._require_live_states(candidate_state_inputs,
                                              candidate.name)
                    candidate_inputs = [
                        resource for resource, _owner in candidate_states
                    ]
                    candidate_zone = (self.resource_zones.get(
                        candidate_inputs[0]) if candidate_inputs else None)
                    if candidate_zone != zone:
                        break
                    if len(candidate_inputs) != 1:
                        raise ValueError(
                            "Lanes v1 measurement requires one physical "
                            "resource per PEG event")
                    if seen.intersection(candidate_inputs):
                        break
                    seen.update(candidate_inputs)
                    self._mark_used(candidate_inputs)
                    measurement = self._instrument_semantic(
                        _text(candidate.attributes["measurement"]))
                    if measurement == "measure_x":
                        self._emit_action("h", candidate_inputs)
                        if self._state_results(candidate):
                            restore_x.extend(candidate_inputs)
                    elif measurement != "measure_z":
                        raise NotImplementedError(
                            "Lanes v1 does not support measurement "
                            f"@{measurement}")
                    if self._state_results(candidate):
                        self._propagate_state_results(candidate,
                                                      candidate_states, states)
                    else:
                        for resource in candidate_inputs:
                            self.current_states[resource] = None
                    cursor += 1
                self._require_exact_zone_effect(seen, zone, "measurement")
                self.lines.extend((
                    f"const_zone 0x{self.zone_indices[zone]:08x}",
                    "measure 1",
                    "await_measure",
                ))
                self._mark_used(
                    resource
                    for resource, current_zone in self.resource_zones.items()
                    if current_zone == zone)
                if restore_x:
                    self._emit_action("h", restore_x)
                continue
            elif name == "phys.call":
                body = operation.regions[0].blocks[0]
                nested = dict(states)
                for argument, source in zip(body.arguments, operation.operands):
                    if _resource(source.type) is not None:
                        state = states[source]
                        nested[argument] = state
                        resource, _owner = state
                        self.current_states[resource] = argument
                returned = self._emit_block(body, nested)
                for result, value in zip(operation.results, returned):
                    if _resource(result.type) is not None:
                        state = nested[value]
                        resource, _owner = state
                        states[result] = state
                        self.current_states[resource] = result
            elif name == "phys.delay":
                self._mark_used(inputs)
                self.lines.append(
                    f"; delay {_text(operation.attributes['duration_ns'])} ns is host scheduled"
                )
                self._propagate_state_results(operation, input_states, states)
            elif name == "phys.barrier":
                domains = tuple(
                    _text(value)
                    for value in operation.attributes.get("domains", ()))
                self._mark_used(self.current_states if "clock" in
                                domains else inputs)
                self.lines.append("barrier")
                self._propagate_state_results(operation, input_states, states)
            elif name == "event.fence":
                self.lines.append("; semantic fence")
            elif name in {"phys.yield", "phys.return"}:
                yielded = tuple(operation.operands)
            elif name == "phys.release":
                for resource in inputs:
                    del self.active_owners[resource]
                    del self.current_states[resource]
            else:
                raise NotImplementedError(
                    f"Lanes v1 cannot faithfully lower {name}")
            cursor += 1
        return yielded

    def emit(self):
        count = len(self.resource_indices)
        if count:
            self.lines.append("")
            self.lines.append("; ---- initial_fill ----")
            for resource in self.resource_indices:
                self.lines.append(
                    f"const_loc 0x{self._location_word(resource):016x}")
            self.lines.append(f"initial_fill {count}")

        self._emit_block(self.graph.regions[0].blocks[0], {})
        self.lines.append("halt")
        return "\n".join(self.lines) + "\n"


def lanes_bytecode_text():

    def finalize(module, ctx, *, root_symbol, **_):
        return _LanesEmitter(module, root_symbol).emit()

    return public_finalizer(finalize, "lanes-bytecode")


__all__ = ["lanes_bytecode_text"]

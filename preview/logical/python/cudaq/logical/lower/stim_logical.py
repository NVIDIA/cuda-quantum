# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Stateful Stim emission for folded logical/Fabric graphs."""

from __future__ import annotations

from dataclasses import replace

from .stim_common import (
    CompiledInterfaceManifest,
    ProjectedMeasurement,
    ProjectedPort,
    _code_name,
    _is_patch_like,
    _pairs,
    _partition,
    _symbol,
    _text,
)


class _Emitter:

    def __init__(self, module):
        self.module = module
        self.symbols = {
            _symbol(view.operation): view.operation
            for view in module.body.operations
            if _symbol(view.operation) is not None
        }
        self.codes = {}
        self.code_has_represented_structure = {}
        for name, operation in self.symbols.items():
            if operation.name != "fabric.code":
                continue
            raw = {
                str(named.name): int(named.attr)
                for named in operation.attributes["partitions"]
            }
            order = [key for key in ("data", "sx", "sz") if key in raw]
            order.extend(key for key in raw if key not in order)
            offset = 0
            partitions = {}
            for key in order:
                partitions[key] = tuple(range(offset, offset + raw[key]))
                offset += raw[key]
            rows = {}
            for key in ("hx", "hz", "gx", "gz", "lx", "lz"):
                rows[key] = tuple(
                    tuple(int(value) for value in row) for row in operation.
                    attributes[key]) if key in operation.attributes else ()
            self.codes[name] = (partitions, rows, offset)
            represented = False
            for key in (
                    "stabilizers",
                    "gauges",
                    "stabilizer_basis",
                    "gauge_x_basis",
                    "gauge_z_basis",
                    "anti_stabilizers",
            ):
                if key not in operation.attributes:
                    continue
                value = operation.attributes[key]
                try:
                    represented = represented or len(value) != 0
                except TypeError:
                    represented = True
            expected_canonical = {
                "logical_x_basis": ((1, 2), (1, 0)),
                "logical_z_basis": ((1, 2), (0, 1)),
                "encoding_clifford": ((2, 2), (1, 0, 0, 1)),
            }
            for key, (shape, bits) in expected_canonical.items():
                if key not in operation.attributes:
                    continue
                value = operation.attributes[key]
                represented = represented or (
                    tuple(value.type.shape) != shape or
                    tuple(int(bit) for bit in value) != bits)
            self.code_has_represented_structure[name] = represented
        self.code_properties = {
            name: {
                key: int(operation.attributes[key])
                for key in ("distance", "n", "k", "r")
                if key in operation.attributes
            }
            for name, operation in self.symbols.items()
            if operation.name == "fabric.code"
        }
        self.lines = []
        self.record_indices = {}
        self.inline_records = {}
        self.projected_measurements = []
        self.measurement_count = 0
        self.next_patch = 0
        self.patch_bases = {}
        self.patch_layouts = {}
        self.patch_transforms = {}
        self.next_qubit = 0
        self.root_symbol = None
        # Data-qubit indices of each patch-typed input port, in interface
        # order, filled in by ``emit``. Analysis backends (the Choi verifier)
        # need the emitter's actual per-port carrier layout — ports are packed
        # by full patch footprint, not by data width — to splice input states
        # onto the right qubits.
        self.input_port_data = ()
        self.interface_manifest = None

    def _new_patch(self, code):
        patch = self.next_patch
        self.next_patch += 1
        base = self.next_qubit
        self.next_qubit += self.codes[code][2]
        self.patch_bases[patch] = (code, base)
        self.patch_layouts[patch] = {
            name: tuple(base + value for value in relative)
            for name, relative in self.codes[code][0].items()
        }
        return patch

    def _qubits(self, patch, partition):
        try:
            if partition == "all":
                return tuple(
                    qubit for values in self.patch_layouts[patch].values()
                    for qubit in values)
            return self.patch_layouts[patch][partition]
        except KeyError as exc:
            code = self.patch_bases[patch][0]
            raise ValueError(
                f"code @{code} has no partition {partition!r}") from exc

    def _require_trivial_preparation(self, patch, operation_name):
        code = self.patch_bases[patch][0]
        partitions, rows, total_carriers = self.codes[code]
        properties = self.code_properties.get(code, {})
        identity_logical = lambda family: family in ((), ((0,),))
        trivial = (properties == {
            "distance": 1,
            "n": 1,
            "k": 1,
            "r": 0
        } and partitions.get("data") == (0,) and total_carriers == 1 and
                   all(not rows[name] for name in ("hx", "hz", "gx", "gz")) and
                   identity_logical(rows["lx"]) and
                   identity_logical(rows["lz"]) and
                   not self.code_has_represented_structure.get(code, False))
        if not trivial:
            raise NotImplementedError(
                f"Stim emission cannot implement {operation_name} for encoded "
                f"code @{code}; reset-based preparation is valid only for a "
                "verified trivial one-carrier code (n=1, k=1, r=0)")

    def _root_ports(self, root):
        if root.name != "fabric.gadget" or "spec" not in root.attributes:
            return ()
        name = _text(root.attributes["spec"])
        spec = self.symbols.get(name)
        if spec is None or spec.name != "fabric.gadget_spec":
            raise ValueError(f"gadget references missing specification @{name}")
        result = []
        for ordinal, encoded in enumerate(spec.attributes.get("ports", ())):
            fields = {str(named.name): named.attr for named in encoded}
            result.append((
                ordinal,
                _text(fields["name"]),
                _text(fields["direction"]),
            ))
        return tuple(result)

    def _projected_port(self, ordinal, name, direction, patch):
        return ProjectedPort(
            ordinal,
            name,
            direction,
            patch,
            tuple((partition, tuple(carriers))
                  for partition, carriers in self.patch_layouts[patch].items()),
        )

    def _record_projected_measurement(
            self,
            *,
            record,
            symbol,
            call_path,
            field,
            lane,
            carriers,
            measurement_index,
            aliases=(),
    ):
        # The compiled interface indexes semantic gadget records in emission
        # order, independently of the textual instruction line count.
        projected_index = len(self.projected_measurements)
        aliases = tuple(aliases)
        if self.root_symbol is not None:
            root_alias = f"{self.root_symbol}.{record}"
            self.record_indices[root_alias] = measurement_index
            aliases = tuple(dict.fromkeys((*aliases, root_alias)))
        self.projected_measurements.append(
            ProjectedMeasurement(
                index=projected_index,
                record=record,
                symbol=symbol,
                call_path=call_path,
                field=field,
                lane=lane,
                carriers=tuple(carriers),
                aliases=aliases,
            ))

    @staticmethod
    def _qualified_record_path(record, instance_path):
        """Qualify records produced inside repeated/called execution scopes."""

        if record.startswith("__qlx_"):
            raise ValueError(
                "measurement record names beginning '__qlx_' are reserved "
                "for compiler-derived invocation qualification")
        return ".".join((*instance_path, record)) if instance_path else record

    def _selected_qubits(self, operation, patch, partition):
        qubits = self._qubits(patch, partition)
        if "indices" not in operation.attributes:
            return qubits
        return tuple(
            qubits[int(index)] for index in operation.attributes["indices"])

    def _encoding_code(self, encoding_name):
        encoding = self.symbols.get(encoding_name)
        if encoding is None or encoding.name != "fabric.encoding":
            raise ValueError(
                f"patch transform references missing encoding @{encoding_name}")
        return _text(encoding.attributes["code"])

    @staticmethod
    def _partition_sizes(attribute):
        raw = {str(named.name): int(named.attr) for named in attribute}
        order = [key for key in ("data", "sx", "sz") if key in raw]
        order.extend(key for key in raw if key not in order)
        return tuple((name, raw[name]) for name in order)

    def _new_frame(self, transform_name, source_patch):
        transform = self.symbols.get(transform_name)
        if transform is None or transform.name != "fabric.patch_transform":
            raise ValueError(
                f"fabric.transform_begin references missing transform @{transform_name}"
            )
        partitions = self._partition_sizes(
            transform.attributes["frame_partitions"])
        frame_size = sum(size for _, size in partitions)
        carriers = [None] * frame_size
        source_data = self._qubits(source_patch, "data")
        source_support = tuple(
            int(value) for value in transform.attributes["source_support"])
        for relative, frame_index in enumerate(source_support):
            carriers[frame_index] = source_data[relative]
        for index, carrier in enumerate(carriers):
            if carrier is None:
                carriers[index] = self.next_qubit
                self.next_qubit += 1

        patch = self.next_patch
        self.next_patch += 1
        source_code = self._encoding_code(_text(transform.attributes["source"]))
        self.patch_bases[patch] = (source_code, 0)
        self.patch_transforms[patch] = transform_name
        offset = 0
        layout = {}
        for name, size in partitions:
            layout[name] = tuple(carriers[offset:offset + size])
            offset += size
        self.patch_layouts[patch] = layout
        return patch

    def _end_frame(self, transform_name, frame_patch):
        transform = self.symbols.get(transform_name)
        if transform is None or transform.name != "fabric.patch_transform":
            raise ValueError(
                f"fabric.transform_end references missing transform @{transform_name}"
            )
        if self.patch_transforms.get(frame_patch) != transform_name:
            raise ValueError(
                "patch frame was committed with the wrong transform")
        destination_code = self._encoding_code(
            _text(transform.attributes["destination"]))
        frame_flat = self._qubits(frame_patch, "all")
        destination_support = tuple(
            int(value) for value in transform.attributes["destination_support"])
        data = tuple(frame_flat[index] for index in destination_support)

        patch = self.next_patch
        self.next_patch += 1
        layout = {"data": data}
        for name, relative in self.codes[destination_code][0].items():
            if name == "data":
                continue
            size = len(relative)
            if name in self.patch_layouts[frame_patch] and len(
                    self.patch_layouts[frame_patch][name]) == size:
                layout[name] = self.patch_layouts[frame_patch][name]
            else:
                layout[name] = tuple(
                    range(self.next_qubit, self.next_qubit + size))
                self.next_qubit += size
        self.patch_bases[patch] = (destination_code, 0)
        self.patch_layouts[patch] = layout
        return patch

    def _gate(self, name, targets, **_options):
        if targets:
            self.lines.append(f"{name} " + " ".join(map(str, targets)))

    def _measure(
            self,
            targets,
            *,
            symbol,
            record,
            call_path,
            record_prefix=(),
            partition=None,
            syndrome=False,
            instruction="M",
    ):
        if not targets:
            return
        self._gate(instruction, targets)
        for local, carrier in enumerate(targets):
            index = self.measurement_count
            self.measurement_count += 1
            if syndrome:
                field = "s"
            else:
                field = partition
            relative = f"{record}.{field}{local}"
            names = (
                f"{symbol}.{relative}",
                f"{symbol}.{record}[{local}]",
            )
            for name in names:
                self.record_indices[name] = index
            self._record_projected_measurement(
                record=self._qualified_record_path(relative, record_prefix),
                symbol=symbol,
                call_path=call_path,
                field=field,
                lane=local,
                carriers=(carrier,),
                measurement_index=index,
                aliases=names,
            )

    def _wire_results(self, operation, patch_values, patches):
        patch_results = [
            value for value in operation.results if _is_patch_like(value.type)
        ]
        if len(patch_results) != len(patches):
            if len(patches) == 1:
                patches = patches * len(patch_results)
            else:
                raise ValueError(
                    f"{operation.name} patch result arity is not ownership-preserving"
                )
        for result, patch in zip(patch_results, patches):
            patch_values[result] = patch

    def _emit_callable(
            self,
            callable_op,
            argument_patches,
            call_stack=(),
            *,
            record_prefix=(),
    ):
        symbol = _symbol(callable_op)
        if symbol in call_stack:
            raise ValueError(f"recursive Fabric call graph through @{symbol}")
        body_owner = callable_op
        if (callable_op.name == "fabric.gadget" and
                "realization" in callable_op.attributes):
            realization = _text(callable_op.attributes["realization"])
            body_owner = self.symbols.get(realization)
            if body_owner is None or body_owner.name != "fabric.circuit":
                raise ValueError(
                    f"gadget @{symbol} references missing realization @{realization}"
                )
        block = body_owner.regions[0].blocks[0]
        patch_values = {
            argument: patch
            for argument, patch in zip(block.arguments, argument_patches)
        }

        def emit_block(block_, local_values, stack, instance_path):
            returned = None
            for operation_ordinal, view in enumerate(block_.operations):
                operation = view.operation
                name = operation.name
                patch_operands = [
                    local_values[value]
                    for value in operation.operands
                    if _is_patch_like(value.type)
                ]
                if name in {"fabric.return", "fabric.protocol_return"}:
                    returned = tuple(patch_operands)
                    continue
                if name == "fabric.alloc":
                    code = _text(operation.attributes["code"])
                    patch = self._new_patch(code)
                    local_values[operation.result] = patch
                    continue
                if name == "fabric.transform_begin":
                    transform_name = _text(operation.attributes["transform"])
                    frame = self._new_frame(transform_name, patch_operands[0])
                    local_values[operation.result] = frame
                    continue
                if name == "fabric.transform_end":
                    transform_name = _text(operation.attributes["transform"])
                    destination = self._end_frame(transform_name,
                                                  patch_operands[0])
                    local_values[operation.result] = destination
                    continue
                if name == "fabric.dealloc":
                    continue
                if name == "fabric.call":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved Fabric call @{callee_name}")
                    results = self._emit_callable(
                        callee,
                        patch_operands,
                        (*stack, symbol),
                        record_prefix=(*instance_path,
                                       f"__qlx_call{operation_ordinal}"),
                    )
                    self._wire_results(operation, local_values, results)
                    continue
                if name == "fabric.relocate":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved trajectory realization @{callee_name}")
                    results = self._emit_callable(
                        callee,
                        patch_operands, (*stack, symbol),
                        record_prefix=(
                            *instance_path,
                            f"__qlx_relocate{operation_ordinal}",
                        ))
                    self._wire_results(operation, local_values, results)
                    continue
                if name == "fabric.establish_support":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved distributed-support realization @{callee_name}"
                        )
                    results = self._emit_callable(
                        callee,
                        patch_operands, (*stack, symbol),
                        record_prefix=(
                            *instance_path,
                            f"__qlx_support{operation_ordinal}",
                        ))
                    self._wire_results(operation, local_values, results)
                    continue
                if name == "fabric.establish_topological_record":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved topological-record realization @{callee_name}"
                        )
                    results = self._emit_callable(
                        callee,
                        patch_operands, (*stack, symbol),
                        record_prefix=(
                            *instance_path,
                            f"__qlx_topological{operation_ordinal}",
                        ))
                    self._wire_results(operation, local_values, results)
                    continue
                if name == "fabric.repeat":
                    count = int(operation.attributes["count"])
                    current = tuple(patch_operands)
                    nested = operation.regions[0].blocks[0]
                    for iteration in range(count):
                        nested_values = {
                            argument: patch for argument, patch in zip(
                                nested.arguments, current)
                        }
                        current = emit_block(
                            nested,
                            nested_values,
                            stack,
                            (*instance_path,
                             f"__qlx_repeat{operation_ordinal}_{iteration}"),
                        ) or current
                    self._wire_results(operation, local_values, current)
                    continue
                if name == "fabric.yield":
                    returned = tuple(patch_operands)
                    continue
                if name in {"fabric.if", "scf.if"}:
                    raise NotImplementedError(
                        "Stim circuit emission cannot erase runtime feedback; "
                        "author a statically resolved P2 circuit")
                if name == "fabric.while":
                    raise NotImplementedError(
                        "Stim circuit emission cannot execute dynamic while; "
                        "author a bounded static repeat")
                if name == "fabric.retry":
                    raise NotImplementedError(
                        "Stim circuit emission cannot erase selection retry; "
                        "lower through a runtime policy or expand a bounded attempt model"
                    )
                if name in {
                        "fabric.encoding_unpack",
                        "fabric.map_children",
                        "fabric.encoding_pack",
                }:
                    raise NotImplementedError(
                        "Stim emission requires hierarchical protocols to be "
                        "lowered or flattened explicitly")
                if name == "fabric.measure_product":
                    self._emit_measure_product(
                        operation,
                        patch_operands,
                        symbol,
                        (*stack, symbol, *instance_path),
                        record_prefix=instance_path,
                    )
                    self._wire_results(operation, local_values, patch_operands)
                    continue
                if name in {
                        "fabric.parity", "fabric.all_false", "fabric.all_zero"
                }:
                    # Classical predicates remain available to P2 selection
                    # lowering but do not add a quantum-circuit instruction.
                    continue
                if name == "fabric.rotate_product":
                    raise NotImplementedError(
                        "Stim cannot evolve arbitrary logical Pauli-product rotations"
                    )
                if name == "fabric.unpack_resource":
                    raise NotImplementedError(
                        "the encoded resource payload has no circuit-level "
                        "producer; select or inline a concrete production "
                        "protocol before circuit emission")
                if name == "fabric.pack_resource":
                    raise NotImplementedError(
                        "circuit emission cannot return an encoded resource "
                        "payload; inline its consumer or use a resource target")
                if name.startswith("fabric.") and patch_operands:
                    partition = (_partition(operation.attributes["partition"])
                                 if "partition" in operation.attributes else
                                 "data")
                    if name in {
                            "fabric.h", "fabric.s", "fabric.sdg", "fabric.x",
                            "fabric.z"
                    }:
                        gate = {
                            "fabric.h": "H",
                            "fabric.s": "S",
                            "fabric.sdg": "S_DAG",
                            "fabric.x": "X",
                            "fabric.z": "Z",
                        }[name]
                        self._gate(
                            gate,
                            self._selected_qubits(operation, patch_operands[0],
                                                  partition),
                        )
                    elif name == "fabric.prep_z":
                        self._require_trivial_preparation(
                            patch_operands[0], name)
                        self._gate(
                            "R",
                            self._qubits(patch_operands[0], "data"),
                        )
                    elif name == "fabric.prep_x":
                        self._require_trivial_preparation(
                            patch_operands[0], name)
                        targets = self._qubits(patch_operands[0], "data")
                        self._gate("R", targets)
                        self._gate("H", targets)
                    elif name == "fabric.reset":
                        self._gate(
                            "R",
                            self._selected_qubits(operation, patch_operands[0],
                                                  partition),
                        )
                    elif name == "fabric.init_basis":
                        targets = self._selected_qubits(operation,
                                                        patch_operands[0],
                                                        partition)
                        self._gate("R", targets)
                        if "x" in str(operation.attributes["basis"]).lower():
                            self._gate("H", targets)
                    elif name in {"fabric.t", "fabric.tdg"}:
                        raise NotImplementedError(
                            "Stim emission is Clifford-only and rejects "
                            f"non-Clifford operation {name}")
                    elif name == "fabric.inject":
                        raise NotImplementedError(
                            "resource injection is a semantic leaf, not a "
                            "transversal physical gate; select a concrete "
                            "gate-teleportation realization before Stim "
                            "emission")
                    elif name in {"fabric.cx", "fabric.cz"}:
                        self._emit_two_patch(operation, patch_operands)
                    elif name == "fabric.transversal_cx":
                        control = self._qubits(patch_operands[0], "data")
                        target = self._qubits(patch_operands[1], "data")
                        if len(control) != len(target):
                            raise ValueError(
                                "fabric.transversal_cx requires equal data widths"
                            )
                        permutation = tuple(
                            int(value) for value in operation.attributes["perm"]
                        ) if "perm" in operation.attributes else tuple(
                            range(len(target)))
                        if len(permutation) != len(target):
                            raise ValueError(
                                "fabric.transversal_cx permutation width disagrees with the target code"
                            )
                        if set(permutation) != set(range(len(target))):
                            raise ValueError(
                                "fabric.transversal_cx perm must be a permutation of target data indices"
                            )
                        self._gate(
                            "CX",
                            tuple(value
                                  for index, control_qubit in enumerate(control)
                                  for value in (control_qubit,
                                                target[permutation[index]])),
                        )
                    elif name in {"fabric.mz", "fabric.measure_basis"}:
                        targets = self._selected_qubits(operation,
                                                        patch_operands[0],
                                                        partition)
                        x_basis = (name == "fabric.measure_basis" and
                                   "x" in str(
                                       operation.attributes["basis"]).lower())
                        start = self.measurement_count
                        self._measure(
                            targets,
                            symbol=symbol,
                            record=(_text(operation.attributes["record"])
                                    if "record" in operation.attributes else
                                    f"measurement{operation_ordinal}"),
                            call_path=(*stack, symbol, *instance_path),
                            record_prefix=instance_path,
                            partition=partition,
                            instruction="MX" if x_basis else "M",
                        )
                        self.inline_records[operation.results[-1]] = tuple(
                            range(start, self.measurement_count))
                    elif name == "fabric.mpp":
                        targets = self._selected_qubits(operation,
                                                        patch_operands[0],
                                                        partition)
                        paulis = _text(operation.attributes["paulis"])
                        # Leading '-' == negated product: Stim complements the
                        # record when the first target is inverted.
                        negated = paulis.startswith("-")
                        paulis = paulis.removeprefix("-")
                        if len(targets) != len(paulis):
                            raise ValueError(
                                "fabric.mpp target and Pauli widths disagree")
                        mpp_targets = [
                            f"{pauli}{qubit}"
                            for pauli, qubit in zip(paulis, targets)
                        ]
                        if negated:
                            mpp_targets[0] = "!" + mpp_targets[0]
                        self.lines.append("MPP " + "*".join(mpp_targets))
                        index = self.measurement_count
                        self.measurement_count += 1
                        record = (_text(operation.attributes["record"])
                                  if "record" in operation.attributes else
                                  f"measurement{operation_ordinal}")
                        aliases = (
                            f"{symbol}.{record}.outcome",
                            f"{symbol}.{record}[0]",
                        )
                        for alias in aliases:
                            self.record_indices[alias] = index
                        self._record_projected_measurement(
                            record=self._qualified_record_path(
                                f"{record}.outcome", instance_path),
                            symbol=symbol,
                            call_path=(*stack, symbol, *instance_path),
                            field="outcome",
                            lane=0,
                            carriers=targets,
                            measurement_index=index,
                            aliases=aliases,
                        )
                        self.inline_records[operation.results[-1]] = (index,)
                    elif name == "fabric.read_syndrome_ancillas":
                        patch = patch_operands[0]
                        targets = self._qubits(patch, "sx") + self._qubits(
                            patch, "sz")
                        start = self.measurement_count
                        self._measure(
                            targets,
                            symbol=symbol,
                            record=(_text(operation.attributes["record"])
                                    if "record" in operation.attributes else
                                    f"syndrome{operation_ordinal}"),
                            call_path=(*stack, symbol, *instance_path),
                            record_prefix=instance_path,
                            syndrome=True,
                        )
                        self.inline_records[operation.results[-1]] = tuple(
                            range(start, self.measurement_count))
                    elif name == "fabric.assemble_syndrome":
                        sx = self.inline_records.get(operation.operands[1])
                        sz = self.inline_records.get(operation.operands[2])
                        if sx is None or sz is None:
                            raise ValueError(
                                "assemble_syndrome references unresolved "
                                "measurement bundles")
                        assembled = (*sx, *sz)
                        record = _text(operation.attributes["record"])
                        for ordinal, measurement in enumerate(assembled):
                            alias = f"{symbol}.{record}.s{ordinal}"
                            self.record_indices[alias] = measurement
                            for index, projected in enumerate(
                                    self.projected_measurements):
                                if any(
                                        self.record_indices.get(existing) ==
                                        measurement
                                        for existing in projected.aliases):
                                    self.projected_measurements[
                                        index] = replace(
                                            projected,
                                            aliases=tuple(
                                                dict.fromkeys(
                                                    (*projected.aliases,
                                                     alias))),
                                        )
                                    break
                        self.inline_records[operation.results[-1]] = assembled
                    elif name == "fabric.permute":
                        data = self._qubits(patch_operands[0], "data")
                        permutation = tuple(
                            int(value)
                            for value in operation.attributes["perm"])
                        seen = set()
                        swaps = []
                        for start in range(len(permutation)):
                            if start in seen:
                                continue
                            seen.add(start)
                            if permutation[start] == start:
                                continue
                            cycle = [start]
                            current = permutation[start]
                            while current not in seen:
                                seen.add(current)
                                cycle.append(current)
                                current = permutation[current]
                            for index in range(len(cycle) - 2, -1, -1):
                                swaps.extend((data[cycle[index]],
                                              data[cycle[index + 1]]))
                        self._gate("SWAP", tuple(swaps))
                    elif name in {"fabric.idle", "fabric.barrier"}:
                        pass
                    else:
                        raise NotImplementedError(
                            f"Stim circuit emission does not support executable operation {name}"
                        )
                    self._wire_results(operation, local_values, patch_operands)
                    continue
                if name.startswith("fabric."):
                    raise NotImplementedError(
                        f"Stim circuit emission does not support executable operation {name}"
                    )
            return returned

        returned = emit_block(block, patch_values, call_stack, record_prefix)
        return tuple(argument_patches) if returned is None else returned

    @staticmethod
    def _multiply_pauli(left, right):
        if left == "I":
            return 1, right
        if right == "I":
            return 1, left
        if left == right:
            return 1, "I"
        table = {
            ("X", "Y"): (1j, "Z"),
            ("Y", "X"): (-1j, "Z"),
            ("X", "Z"): (-1j, "Y"),
            ("Z", "X"): (1j, "Y"),
            ("Y", "Z"): (1j, "X"),
            ("Z", "Y"): (-1j, "X"),
        }
        return table[(left, right)]

    def _emit_measure_product(
            self,
            operation,
            patches,
            symbol,
            call_path,
            *,
            record_prefix=(),
    ):
        patch_indices = [
            int(value) for value in operation.attributes["patch_indices"]
        ]
        logical_indices = [
            int(value) for value in operation.attributes["logical_indices"]
        ]
        encoded = _text(operation.attributes["pauli_product"])
        sign = -1 if encoded.startswith("-") else 1
        paulis = encoded.removeprefix("-")
        physical = {}
        phase = complex(sign)

        def multiply_string(patch, support, pauli):
            nonlocal phase
            data = self._qubits(patch, "data")
            for relative in support:
                qubit = data[relative]
                factor, result = self._multiply_pauli(physical.get(qubit, "I"),
                                                      pauli)
                phase *= factor
                if result == "I":
                    physical.pop(qubit, None)
                else:
                    physical[qubit] = result

        for patch_index, logical, pauli in zip(patch_indices, logical_indices,
                                               paulis):
            patch = patches[patch_index]
            code = self.patch_bases[patch][0]
            rows = self.codes[code][1]
            x_basis = (*rows["lx"], *rows["gx"])
            z_basis = (*rows["lz"], *rows["gz"])
            if logical >= len(x_basis) or logical >= len(z_basis):
                raise ValueError(
                    f"code @{code} has no logical representative {logical}")
            if pauli == "X":
                multiply_string(patch, x_basis[logical], "X")
            elif pauli == "Z":
                multiply_string(patch, z_basis[logical], "Z")
            elif pauli == "Y":
                phase *= 1j
                multiply_string(patch, x_basis[logical], "X")
                multiply_string(patch, z_basis[logical], "Z")
            else:
                raise ValueError(f"invalid logical Pauli factor {pauli!r}")
        if abs(phase.imag) > 1e-9 or abs(abs(phase.real) - 1.0) > 1e-9:
            raise ValueError(
                "logical Pauli product did not map to a Hermitian observable")
        targets = [
            f"{pauli}{qubit}" for qubit, pauli in sorted(physical.items())
        ]
        if not targets:
            raise ValueError("logical Pauli product reduced to identity")
        if phase.real < 0:
            targets[0] = "!" + targets[0]
        self.lines.append("MPP " + "*".join(targets))
        index = self.measurement_count
        self.measurement_count += 1
        self.inline_records[operation.results[-1]] = (index,)
        record = _text(operation.attributes["record"]
                      ) if "record" in operation.attributes else "mpp"
        aliases = (f"{symbol}.{record}", f"{symbol}.{record}.outcome")
        for alias in aliases:
            self.record_indices[alias] = index
        self._record_projected_measurement(
            record=self._qualified_record_path(f"{record}.outcome",
                                               record_prefix),
            symbol=symbol,
            call_path=call_path,
            field="outcome",
            lane=0,
            carriers=tuple(physical),
            measurement_index=index,
            aliases=aliases,
        )

    def _emit_two_patch(self, operation, patches):
        ctrl = _partition(operation.attributes["ctrl"])
        targ = _partition(operation.attributes["targ"])
        left = self._qubits(patches[0], ctrl)
        right_patch = patches[-1]
        right = self._qubits(right_patch, targ)
        pairs = None
        if "pairs" in operation.attributes:
            parsed = _pairs(operation.attributes["pairs"], len(left),
                            len(right))
            pairs = tuple((left[i], right[j]) for i, j in parsed)
        elif "schedule" in operation.attributes:
            schedule = _text(operation.attributes["schedule"]).lower()
            code = self.patch_bases[patches[0]][0]
            rows = self.codes[code][1][schedule]
            if schedule == "hx":
                pairs = tuple(
                    (left[check], self._qubits(right_patch, "data")[data])
                    for check, row in enumerate(rows)
                    for data in row)
            elif schedule == "hz":
                pairs = tuple(
                    (self._qubits(patches[0], "data")[data], right[check])
                    for check, row in enumerate(rows)
                    for data in row)
        if pairs is None:
            if len(left) != len(right):
                raise ValueError(
                    f"{operation.name} needs equal partitions, pairs=, or a CSS schedule"
                )
            pairs = tuple(zip(left, right))
        flat = [value for pair in pairs for value in pair]
        gate = "CX" if operation.name == "fabric.cx" else "CZ"
        self._gate(gate, flat)

    def emit(self, root_symbol):
        root = self.symbols.get(root_symbol)
        if root is None:
            raise ValueError(f"missing root @{root_symbol}")
        if root.name != "fabric.gadget":
            raise TypeError(
                "Stim circuit emission requires a legalized entry gadget")
        self.root_symbol = _symbol(root)
        arguments = []
        for argument in root.regions[0].blocks[0].arguments:
            code = _code_name(argument.type)
            if code is not None:
                arguments.append(self._new_patch(code))
        self.input_port_data = tuple(
            self.patch_layouts[patch]["data"] for patch in arguments)
        returned = self._emit_callable(
            root,
            tuple(arguments),
        )
        ports = self._root_ports(root)
        if ports:
            if len(ports) != len(arguments):
                raise ValueError(
                    "gadget specification port count disagrees with projected arguments"
                )
            input_bindings = tuple(
                (ordinal, name, direction, patch)
                for (ordinal, name,
                     direction), patch in zip(ports, arguments, strict=True)
                if direction in {"input", "inout"})
            output_ports = tuple(
                port for port in ports if port[2] in {"output", "inout"})
            if len(output_ports) != len(returned):
                raise ValueError(
                    "gadget specification output count disagrees with returned patches"
                )
            output_bindings = tuple(
                (*port, patch)
                for port, patch in zip(output_ports, returned, strict=True))
        else:
            # Protocols do not yet carry gadget_spec ports. Preserve their
            # concrete argument/return boundary without pretending those
            # fallback names are a reusable semantic ABI.
            input_bindings = tuple((ordinal, f"arg{ordinal}", "inout", patch)
                                   for ordinal, patch in enumerate(arguments))
            output_bindings = tuple(
                (ordinal, f"result{ordinal}", "inout", patch)
                for ordinal, patch in enumerate(returned))
        self.interface_manifest = CompiledInterfaceManifest(
            inputs=tuple(
                self._projected_port(ordinal, name, direction, patch)
                for ordinal, name, direction, patch in input_bindings),
            outputs=tuple(
                self._projected_port(ordinal, name, direction, patch)
                for ordinal, name, direction, patch in output_bindings),
            measurements=tuple(self.projected_measurements),
            carrier_count=self.next_qubit,
        )
        return "\n".join(self.lines) + ("\n" if self.lines else "")


__all__ = ["_Emitter"]

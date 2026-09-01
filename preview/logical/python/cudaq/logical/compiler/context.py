# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .. import ir as mlir_ir

from ..programs.definition import DefinitionHandle

_symbol_re = re.compile(r"[^A-Za-z0-9_.$-]")


def _symbol(name: str) -> str:
    value = _symbol_re.sub("_", name)
    if not value or value[0].isdigit():
        value = f"_{value}"
    return value


def _symbol_path(attribute) -> tuple[str, ...]:
    value = getattr(attribute, "value", None)
    if isinstance(value, (tuple, list)):
        return tuple(str(item).lstrip("@") for item in value)
    text = str(attribute if value is None else value).strip('"')
    return tuple(item.lstrip("@") for item in text.split("::") if item)


class CompilationContext:
    """Private, transaction-local linker and MLIR owner."""

    def __init__(self, module=None) -> None:
        if module is None:
            self.context = mlir_ir.Context()
            self.location = mlir_ir.Location.unknown(context=self.context)
            self.module = mlir_ir.Module.create(self.location)
        else:
            if not isinstance(module, mlir_ir.Module):
                raise TypeError("module= must be a cudaq.logical.ir.Module")
            self.module = module
            self.context = module.context
            self.location = mlir_ir.Location.unknown(context=self.context)
        self._definitions: dict[int, DefinitionHandle[Any]] = {}
        self._materialization_traces: list[dict[int, object]] = []
        self._materialization_closures: dict[int, tuple[object, ...]] = {}
        self._definition_types: dict[int, tuple[tuple[Any, ...],
                                                tuple[Any, ...], str]] = {}
        self._value_groups: dict[int, dict[str, int]] = {}
        self._symbols: dict[str, object] = {}
        self._symbol_operations: dict[str, object] = {}
        self._objectives: dict[tuple[Any, ...], str] = {}
        # Exact Boolean-result provenance for protocol calls materialized in
        # this private transaction.  This lets a closed protocol attempt expose
        # a success predicate derived unchanged from a selected gadget profile
        # without inventing a second protocol-analysis object model.
        self._protocol_result_provenance: dict[str, tuple[Any | None, ...]] = {}
        self._protocol_payload_blocks: dict[int, tuple[str, ...]] = {}
        self._resolving: set[int] = set()
        self._resource_streams: (dict[str, tuple[tuple[str, str], ...]] |
                                 None) = None
        self._configure_module()
        for operation in self.walk():
            if "sym_name" not in operation.attributes:
                continue
            attr = operation.attributes["sym_name"]
            name = str(getattr(attr, "value", attr)).strip('"')
            self._symbols.setdefault(name, object())
            self._symbol_operations.setdefault(name, operation)

    @classmethod
    def replay(cls, build) -> "CompilationContext":
        """Open an immutable build in a fresh private linking transaction."""
        self = cls.__new__(cls)
        self.context = mlir_ir.Context()
        self.module = mlir_ir.Module.parse(build.to_mlir(), self.context)
        self.location = mlir_ir.Location.unknown(context=self.context)
        self._definitions = {}
        self._materialization_traces = []
        self._materialization_closures = {}
        self._definition_types = {}
        self._value_groups = {}
        self._symbols = {}
        self._symbol_operations = {}
        self._objectives = {}
        self._protocol_result_provenance = {}
        self._protocol_payload_blocks = {}
        self._resolving = set()
        self._resource_streams = None
        for operation in self.walk():
            if "sym_name" not in operation.attributes:
                continue
            attr = operation.attributes["sym_name"]
            name = str(getattr(attr, "value", attr)).strip('"')
            self._symbols.setdefault(name, object())
            self._symbol_operations.setdefault(name, operation)
        return self

    def walk(self):

        def visit(operation):
            yield operation
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        yield from visit(child.operation)

        yield from visit(self.module.operation)

    def bind_resource_streams(self, device, *, logical_symbol=None) -> None:
        """Bind portable Fabric request placeholders to one selected device."""

        logical_symbol = logical_symbol or device.logical.name
        grouped = {}
        for stream in device.logical.streams:
            grouped.setdefault(stream.produces.name, []).append(
                (logical_symbol, stream.name))
        routes = {kind: tuple(paths) for kind, paths in grouped.items()}
        self._resource_streams = routes

        for operation in self.walk():
            if operation.name != "fabric.resource_request":
                continue
            kind_attr = operation.attributes["kind"]
            kind = str(getattr(kind_attr, "value", kind_attr)).strip('"')
            candidates = routes.get(kind, ())
            if not candidates:
                raise ValueError(
                    f"selected device has no stream producing {kind!r}")
            if len(candidates) != 1:
                raise ValueError(
                    "selected device has ambiguous streams producing "
                    f"{kind!r}: {candidates!r}")
            expected = candidates[0]
            current = _symbol_path(operation.attributes["stream"])
            if len(current) > 1 and current != expected:
                raise ValueError(
                    f"resource request for {kind!r} names stream {current!r}, "
                    f"but the selected device routes it through {expected!r}")
            with self.context:
                operation.attributes["stream"] = mlir_ir.SymbolRefAttr.get(
                    list(expected), context=self.context)

    def find_symbol(
        self,
        name: str,
        operation_name: str | None = None,
        *,
        scan: bool = True,
    ):
        cached = self._symbol_operations.get(name)
        if cached is not None:
            try:
                cached_name = self._operation_symbol(cached)
                cached_operation_name = cached.name
            except RuntimeError:
                # MLIR invalidates an operation wrapper after erase(). Never
                # return that stale wrapper from this transaction-local index.
                self._symbol_operations.pop(name, None)
                self._symbols.pop(name, None)
            else:
                if cached_name != name:
                    # Keep the index coherent if a transform renamed the
                    # declaration through the underlying MLIR API.
                    self._symbol_operations.pop(name, None)
                    self._symbols.pop(name, None)
                    if cached_name is not None:
                        self._symbol_operations.setdefault(cached_name, cached)
                        self._symbols.setdefault(cached_name, object())
                else:
                    return (cached if operation_name is None or
                            cached_operation_name == operation_name else None)
        if not scan:
            return None
        for operation in self.walk():
            if "sym_name" not in operation.attributes:
                continue
            attr = operation.attributes["sym_name"]
            candidate = str(getattr(attr, "value", attr)).strip('"')
            self._symbol_operations.setdefault(candidate, operation)
            if candidate == name:
                return (operation if operation_name is None or
                        operation.name == operation_name else None)
        return None

    def _index_symbol(self, operation, expected: str | None = None) -> str:
        """Record one compiler-created symbol without rescanning the module."""

        symbol = self._operation_symbol(operation)
        if symbol is None:
            raise ValueError(
                "only symbol operations may enter the symbol index")
        if expected is not None and symbol != expected:
            raise ValueError(
                f"symbol operation @{symbol} does not match handle @{expected}")
        self._symbols.setdefault(symbol, object())
        self._symbol_operations[symbol] = operation
        return symbol

    def _configure_module(self) -> None:
        string = mlir_ir.StringAttr.get
        if "qlx.model_version" not in self.module.operation.attributes:
            self.module.operation.attributes["qlx.model_version"] = string(
                "0.3.10-proposed", context=self.context)
        if "qlx.ir_version" not in self.module.operation.attributes:
            self.module.operation.attributes["qlx.ir_version"] = string(
                "0.4-draft", context=self.context)
        if "qlx.profiles" not in self.module.operation.attributes:
            self.module.operation.attributes[
                "qlx.profiles"] = mlir_ir.ArrayAttr.get([],
                                                        context=self.context)
        if "qlx.stages" not in self.module.operation.attributes:
            self.module.operation.attributes[
                "qlx.stages"] = mlir_ir.ArrayAttr.get([], context=self.context)
        if "qlx.facets" not in self.module.operation.attributes:
            self.module.operation.attributes[
                "qlx.facets"] = mlir_ir.ArrayAttr.get([], context=self.context)

    def unique_symbol(self, requested: str) -> str:
        base = _symbol(requested)
        candidate = base
        suffix = 1
        while candidate in self._symbols:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self._symbols[candidate] = object()
        return candidate

    def objective(
        self,
        *,
        family: str,
        name: str,
        inputs: tuple[Any, ...],
        results: tuple[Any, ...],
        semantics: Any = None,
    ) -> str:
        key = (family, name, tuple(map(str, inputs)), tuple(map(str, results)))
        if key in self._objectives:
            return self._objectives[key]
        symbol = self.declare_objective(
            family=family,
            requested_symbol=f"qlx_standard_{name}",
            kind=name,
            inputs=inputs,
            results=results,
            semantics=semantics,
        )
        self._objectives[key] = symbol
        return symbol

    def declare_objective(
        self,
        *,
        family: str,
        requested_symbol: str,
        kind: str,
        inputs: tuple[Any, ...],
        results: tuple[Any, ...],
        semantics: Any = None,
    ) -> str:
        symbol = self.unique_symbol(requested_symbol)
        function_type = mlir_ir.FunctionType.get(inputs,
                                                 results,
                                                 context=self.context)
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(function_type)
        attrs: dict[str, Any] = {
            "sym_name": mlir_ir.StringAttr.get(symbol, context=self.context),
            "function_type": function_type_attr,
            "kind": mlir_ir.StringAttr.get(kind, context=self.context),
        }
        if family == "action":
            from ..algebra.clifford import (
                CliffordAction,
                NonCliffordAction,
            )

            try:
                action = CliffordAction.standard(
                    kind,
                    len(inputs),
                    ports=tuple(f"q{index}" for index in range(len(inputs))),
                )
            except NonCliffordAction:
                pass
            else:
                attrs["clifford_action"] = action.to_mlir_attr(self.context)
        if semantics is not None:
            attrs["semantics"] = semantics
        op_name = "qlx.action" if family == "action" else "qlx.instrument_decl"
        with self.location:
            operation = mlir_ir.Operation.create(op_name,
                                                 results=[],
                                                 operands=[],
                                                 attributes=attrs,
                                                 loc=self.location)
            self.module.body.append(operation)
        self._symbol_operations[symbol] = operation
        return symbol

    def materialize(self, definition):
        """Materialize a definition and retain its transaction-local closure."""

        identity = id(definition)
        previous = self._materialization_closures.get(identity, ())
        for trace in self._materialization_traces:
            trace.setdefault(identity, definition)
            for dependency in previous:
                trace.setdefault(id(dependency), dependency)
        trace = {identity: definition}
        for dependency in previous:
            trace.setdefault(id(dependency), dependency)
        self._materialization_traces.append(trace)
        try:
            handle = self._materialize_impl(definition)
        finally:
            popped = self._materialization_traces.pop()
            assert popped is trace
        self._materialization_closures[identity] = tuple(trace.values())
        return handle

    def materialize_with_dependencies(self, definition):
        """Return a handle and every definition reached while producing it."""

        handle = self.materialize(definition)
        return handle, self._materialization_closures[id(definition)]

    def _materialize_impl(self, definition):
        existing = self.lookup(definition)
        if existing is not None:
            return existing
        snapshot = getattr(definition, "_qlx_direct_snapshot", None)
        if snapshot is not None:
            return self._import_direct_snapshot(definition, snapshot)
        from ..codes import (
            Code,
            CodeProfile,
            Encoding,
            EncodingEpoch,
            EncodingEpochSchema,
            EncodingHierarchy,
            EncodingProjection,
            PatchTransform,
        )
        from ..gadgets import GadgetDefinition
        from ..protocols.definition import ProtocolDefinition
        from ..devices.definition import Device
        from ..qec.lowering import QECLowering

        if isinstance(
                definition,
            (
                Code,
                CodeProfile,
                Encoding,
                EncodingEpoch,
                EncodingEpochSchema,
                EncodingHierarchy,
                EncodingProjection,
                PatchTransform,
            ),
        ):
            from .qec import materialize_qec

            return materialize_qec(self, definition)
        if isinstance(definition, Device):
            from ..architecture.builder import materialize_device

            return materialize_device(self, definition)
        if isinstance(definition, QECLowering):
            from .lowering import materialize_qec_lowering

            return materialize_qec_lowering(self, definition)
        if isinstance(definition, GadgetDefinition):
            from ..gadgets.builder import GadgetBuilder

            identity = id(definition)
            if identity in self._resolving:
                raise RuntimeError(
                    f"recursive definition cycle while materializing "
                    f"{definition.name!r}")
            self._resolving.add(identity)
            try:
                builder = GadgetBuilder(self, definition)
                builder.trace()
                self._definition_types[identity] = (
                    builder.input_types,
                    builder.result_types,
                    builder.symbol,
                )
                handle = DefinitionHandle(symbol=builder.symbol,
                                          kind="gadget",
                                          profile="p2a")
                self.bind(definition, handle)
                self._index_symbol(builder.operation, handle.symbol)
                self.add_profile("p2a")
                return handle
            finally:
                self._resolving.remove(identity)
        if isinstance(definition, ProtocolDefinition):
            from ..protocols.builder import ProtocolBuilder

            identity = id(definition)
            if identity in self._resolving:
                raise RuntimeError(
                    f"recursive definition cycle while materializing "
                    f"{definition.name!r}")
            self._resolving.add(identity)
            try:
                builder = ProtocolBuilder(self, definition)
                builder.trace()
                self._definition_types[identity] = (
                    builder.input_types,
                    builder.result_types,
                    builder.symbol,
                )
                handle = DefinitionHandle(symbol=builder.symbol,
                                          kind="protocol",
                                          profile="p2n")
                self.bind(definition, handle)
                self._index_symbol(builder.operation, handle.symbol)
                self.add_profile("p2n")
                return handle
            finally:
                self._resolving.remove(identity)
        identity = id(definition)
        if identity in self._resolving:
            raise RuntimeError(
                f"recursive definition cycle while materializing {definition.name!r}"
            )
        self._resolving.add(identity)
        try:
            from ..programs.builder import UnplacedBuilder

            builder = UnplacedBuilder(self, definition)
            builder.trace()
            self._definition_types[identity] = (
                builder.input_types,
                builder.result_types,
                builder.symbol,
            )
            self._value_groups[identity] = dict(builder.value_groups)
            if definition.kind == "program":
                handle = DefinitionHandle(symbol=builder.symbol,
                                          kind="program",
                                          profile="p0")
            elif definition.kind == "objective":
                family = builder.objective_family()
                objective_symbol = self.declare_objective(
                    family=family,
                    requested_symbol=definition.name,
                    kind="composite",
                    inputs=builder.input_types,
                    results=builder.result_types,
                    semantics=mlir_ir.FlatSymbolRefAttr.get(
                        builder.symbol, context=self.context),
                )
                if family == "action":
                    from ..algebra.clifford import (
                        CliffordAction,
                        NonCliffordAction,
                    )

                    try:
                        clifford = CliffordAction.from_mlir_program(
                            builder.operation,
                            ports=tuple(definition.signature.parameters),
                        )
                    except NonCliffordAction:
                        pass
                    else:
                        objective = self.find_symbol(objective_symbol,
                                                     "qlx.action")
                        objective.attributes["clifford_action"] = (
                            clifford.to_mlir_attr(self.context))
                handle = DefinitionHandle(
                    symbol=objective_symbol,
                    kind=family,
                    profile="p0",
                )
            else:
                raise TypeError(
                    f"unsupported P0 definition kind {definition.kind!r}")
            self.bind(definition, handle)
            return handle
        finally:
            self._resolving.remove(identity)

    @staticmethod
    def _operation_symbol(operation) -> str | None:
        try:
            attribute = operation.attributes["sym_name"]
        except KeyError:
            return None
        return str(getattr(attribute, "value", attribute)).strip('"')

    @staticmethod
    def _same_symbol_definition(left, right) -> bool:
        """Compare semantic declaration payloads independent of authored labels."""

        if left.name != right.name:
            return False
        ignored = {"sym_name"}
        if left.name == "fabric.encoding":
            # This generated child symbol is determined by the canonical schema
            # of the encoding and is remapped independently immediately after
            # its parent declaration.
            ignored.add("initial_epoch")
        left_names = tuple(
            sorted(name for name in left.attributes if name not in ignored))
        right_names = tuple(
            sorted(name for name in right.attributes if name not in ignored))
        if left_names != right_names:
            return False
        if any(
                str(left.attributes[name]) != str(right.attributes[name])
                for name in left_names):
            return False
        if len(left.regions) != len(right.regions):
            return False
        if not left.regions:
            return True
        # Record identifiers and other body-local strings may be qualified by
        # the owning root label and are not MLIR symbol uses. Region-bearing
        # roots therefore coalesce only under the same label and exact printed
        # body; declarations in their closure remain transaction-wide.
        if (CompilationContext._operation_symbol(left)
                != CompilationContext._operation_symbol(right)):
            return False
        return str(left) == str(right)

    def _import_direct_snapshot(self, definition, build):
        """Link a finished advanced-builder definition into this transaction.

        Direct builders author the canonical IR immediately, so replaying their
        placeholder Python provider would be both lossy and surprising.  The
        frozen build is instead parsed into this transaction's MLIR context.
        Equal symbol definitions are coalesced; a same-name structural conflict
        fails rather than being silently renamed and invalidating references.
        An archival ``qlx.experiment`` is never linked as a semantic
        dependency.  Experiment metadata lives in the consuming Build bundle.
        """

        from .build import Build

        if not isinstance(build, Build):
            raise TypeError(
                "direct definition snapshot must be a CUDA-Q Logical Build")
        imported = mlir_ir.Module.parse(build.to_mlir(), self.context)
        profile_attr = (imported.operation.attributes["qlx.profiles"] if
                        "qlx.profiles" in imported.operation.attributes else ())
        profiles = tuple(
            str(getattr(value, "value", value)).strip('"')
            for value in profile_attr)
        imported_views = tuple(view for view in imported.body.operations
                               if view.operation.name != "qlx.experiment")
        imported_symbols = {
            symbol for view in imported_views
            if (symbol := self._operation_symbol(view.operation)) is not None
        }
        reserved_symbols = {*self._symbols, *imported_symbols}
        root_symbol = build.root.symbol

        def fresh_import_symbol(requested: str) -> str:
            base = _symbol(requested)
            candidate = base
            suffix = 1
            while candidate in reserved_symbols:
                candidate = f"{base}_{suffix}"
                suffix += 1
            reserved_symbols.add(candidate)
            self._symbols[candidate] = object()
            return candidate

        renamed_symbols = {}

        def remap_type(type_):
            text = str(type_)
            for source, target in renamed_symbols.items():
                text = re.sub(
                    rf"@{re.escape(source)}(?![A-Za-z0-9_.$-])",
                    f"@{target}",
                    text,
                )
            return type_ if text == str(type_) else mlir_ir.Type.parse(
                text, context=self.context)

        def remap_value_types(operation):
            for result in operation.results:
                result.set_type(remap_type(result.type))
            for region in operation.regions:
                for block in region.blocks:
                    for argument in block.arguments:
                        argument.set_type(remap_type(argument.type))
                    for child in block.operations:
                        remap_value_types(child.operation)

        def remap_import_symbol(source: str, target: str) -> None:
            nonlocal root_symbol
            mlir_ir.SymbolTable.replace_all_symbol_uses(source, target,
                                                        imported.operation)
            if root_symbol == source:
                root_symbol = target
            renamed_symbols[source] = target
            remap_value_types(imported.operation)

        def equivalent_definition(operation):
            for current in self.walk():
                if self._same_symbol_definition(current, operation):
                    return current
            return None

        linked_views = []
        for view in imported_views:
            operation = view.operation
            symbol = self._operation_symbol(operation)
            if symbol is None:
                raise ValueError(
                    "direct definition snapshots may only contain module-level "
                    f"symbols; found {operation.name}")
            equivalent = equivalent_definition(operation)
            if equivalent is not None:
                canonical = self._operation_symbol(equivalent)
                if canonical != symbol:
                    remap_import_symbol(symbol, canonical)
                continue
            if self.find_symbol(symbol) is not None:
                renamed = fresh_import_symbol(symbol)
                remap_import_symbol(symbol, renamed)
                mlir_ir.SymbolTable.set_symbol_name(operation, renamed)
                symbol = renamed
            linked_views.append((view, symbol))

        # Keep the complete imported closure attached until every conflicting
        # symbol and symbol-bearing type has been rewritten. Detaching an
        # earlier encoding before a later epoch rename would leave its
        # ``initial_epoch`` reference stale.
        for view, symbol in linked_views:
            view.detach_from_parent()
            self.module.body.append(view)
            self._symbols[symbol] = (definition
                                     if symbol == root_symbol else object())

        root = self.find_symbol(root_symbol)
        if root is None:
            raise ValueError(
                f"direct definition snapshot is missing root @{root_symbol}")
        function_type_attr = (root.attributes["function_type"]
                              if "function_type" in root.attributes else None)
        if function_type_attr is None:
            raise ValueError(
                f"direct definition root @{root_symbol} has no function_type")
        function_type = mlir_ir.TypeAttr(function_type_attr).value
        self._definition_types[id(definition)] = (
            tuple(function_type.inputs),
            tuple(function_type.results),
            root_symbol,
        )
        for profile in profiles:
            self.add_profile(profile)
        handle = DefinitionHandle(
            root_symbol,
            build.root.kind,
            build.root.profile,
        )
        self.bind(definition, handle)
        return handle

    def bind(self, definition: object, handle: DefinitionHandle[Any]) -> None:
        self._definitions[id(definition)] = handle

    def bind_existing_protocol(self, definition: object, symbol: str):
        """Bind one retained, already-verified protocol to its Python owner."""

        operation = self.find_symbol(symbol, "fabric.protocol")
        if operation is None:
            raise ValueError(
                f"retained protocol provenance references missing @{symbol}")
        function_type = mlir_ir.TypeAttr(
            operation.attributes["function_type"]).value
        self._definition_types[id(definition)] = (
            tuple(function_type.inputs),
            tuple(function_type.results),
            symbol,
        )
        handle = DefinitionHandle(symbol, "protocol", "p2n")
        self.bind(definition, handle)
        return handle

    def bind_protocol_payload_blocks(self, definition, blocks) -> None:
        blocks = tuple(blocks)
        if (not blocks or any(
                not isinstance(block, str) or not block for block in blocks) or
                len(set(blocks)) != len(blocks)):
            raise ValueError(
                "protocol payload blocks must be unique selected QEC identities"
            )
        self._protocol_payload_blocks[id(definition)] = blocks

    def protocol_payload_blocks(self, definition):
        return self._protocol_payload_blocks.get(id(definition))

    def add_profile(self, profile: str) -> None:
        """Add an implementation product and its canonical stage/facets.

        ``qlx.profiles`` remains a temporary native-compatibility mirror. New
        consumers must use ``qlx.stages`` and ``qlx.facets``.
        """
        from ..stages import stage_and_facets

        stage, facets = stage_and_facets(profile)
        if stage is not None:
            self.add_stage(stage.value)
        for facet in facets:
            self.add_facet(facet.value)
        existing = [
            attr.value
            for attr in self.module.operation.attributes["qlx.profiles"]
        ]
        if profile in existing:
            return
        self.module.operation.attributes[
            "qlx.profiles"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=self.context)
                    for value in (*existing, profile)
                ],
                context=self.context,
            )

    def _add_module_set_value(self, attribute: str, value: str) -> None:
        existing = [
            item.value for item in self.module.operation.attributes[attribute]
        ]
        if value in existing:
            return
        self.module.operation.attributes[attribute] = mlir_ir.ArrayAttr.get(
            [
                mlir_ir.StringAttr.get(item, context=self.context)
                for item in (*existing, value)
            ],
            context=self.context,
        )

    def add_stage(self, stage: str) -> None:
        self._add_module_set_value("qlx.stages", stage)

    def add_facet(self, facet: str) -> None:
        self._add_module_set_value("qlx.facets", facet)

    def lookup(self, definition: object) -> DefinitionHandle[Any] | None:
        return self._definitions.get(id(definition))

    def signature_of(self, definition: object):
        try:
            return self._definition_types[id(definition)]
        except KeyError as exc:
            raise KeyError(
                "definition has not been materialized in this transaction"
            ) from exc

    def value_groups_of(self, definition: object) -> dict[str, int]:
        return dict(self._value_groups.get(id(definition), {}))

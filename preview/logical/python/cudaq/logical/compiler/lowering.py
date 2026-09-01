# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

from .. import ir as mlir_ir

from ..programs.definition import DefinitionHandle
from ..codes import Encoding
from ..qec.lowering import QECLowering
from ..programs.definition import ProgramDefinition
from ..std import LogicalActionRef, LogicalInstrumentRef


@dataclass(frozen=True, slots=True)
class _LinkedDefinitions:
    definitions: tuple[object, ...]
    ordinary_ids: frozenset[int]


def discover_linked_definitions(source_modules, device):
    """Discover CUDA-Q Logical exports from only the root and selected-device modules.

    This is the Python module-linking boundary, not a process-global registry:
    unrelated ``sys.modules`` state is never scanned and no candidate winner is
    cached outside the private compilation transaction.
    """
    from ..codes import (
        Code,
        CodeProfile,
        EncodingHierarchy,
        EncodingProjection,
    )
    from ..gadgets import GadgetDefinition
    from ..protocols.definition import ProtocolDefinition

    definition_types = (
        Code,
        CodeProfile,
        Encoding,
        EncodingHierarchy,
        EncodingProjection,
        GadgetDefinition,
        ProtocolDefinition,
        QECLowering,
    )
    names = [*source_modules]
    if getattr(device, "source_module", None):
        names.append(device.source_module)
    seen_modules: set[str] = set()
    seen_definitions: set[int] = set()
    ordinary_definition_ids: set[int] = set()
    definitions = []

    def append_definition(value, *, ordinary: bool) -> None:
        if ordinary:
            ordinary_definition_ids.add(id(value))
        if id(value) not in seen_definitions:
            seen_definitions.add(id(value))
            definitions.append(value)

    def visit(module: ModuleType,
              *,
              root_prefix: str,
              library: bool = False) -> None:
        if module.__name__ in seen_modules or module.__name__ == "cudaq.logical":
            return
        if module.__name__.startswith("qlx.") and not library:
            return
        seen_modules.add(module.__name__)
        for _, value in sorted(vars(module).items()):
            if isinstance(value, definition_types):
                append_definition(value, ordinary=True)
                continue
            if not isinstance(value, ModuleType):
                continue
            if library:
                # An explicitly linked library module contributes only its
                # own definitions; it never expands transitively.
                continue
            child = value.__name__
            if child == root_prefix or child.startswith(root_prefix + "."):
                visit(value, root_prefix=root_prefix)
            elif child != "cudaq.logical" and child.startswith(
                    "cudaq.logical."):
                # Binding a specific cudaq.logical library submodule in a linked user
                # module (``from cudaq.logical.qec import steane``) is the explicit
                # link act: its definitions become selection candidates.
                # Bare ``import cudaq.logical`` never links the standard library.
                visit(value, root_prefix=child, library=True)

    for name in dict.fromkeys(names):
        module = importlib.import_module(name)
        root_prefix = name.split(".", 1)[0]
        visit(module, root_prefix=root_prefix)
    for binding in getattr(device, "logical_to_qec", ()):
        architecture = getattr(binding, "architecture", None)
        if architecture is None:
            continue
        for definition in architecture.link_roots:
            append_definition(definition, ordinary=False)
    return _LinkedDefinitions(
        definitions=tuple(definitions),
        ordinary_ids=frozenset(ordinary_definition_ids),
    )


def _dict(context, values):
    if not values:
        return None
    return mlir_ir.DictAttr.get(
        {
            str(key): mlir_ir.StringAttr.get(str(value), context=context)
            for key, value in values.items()
        },
        context=context,
    )


def materialize_qec_lowering(transaction, definition: QECLowering):
    existing = transaction.lookup(definition)
    if existing is not None:
        return existing
    context = transaction.context
    symbol = transaction.unique_symbol(definition.name)
    code_refs = []
    for value in definition.codes:
        code_refs.append(transaction.materialize(value).symbol)
    dependency_refs = []
    for dependency in definition.dependencies:
        if hasattr(dependency, "materialize") or hasattr(
                dependency, "provider"):
            dependency_refs.append(transaction.materialize(dependency).symbol)
        elif hasattr(dependency, "name"):
            dependency_refs.append(dependency.name)
        elif isinstance(dependency, str):
            dependency_refs.append(dependency)
        else:
            raise TypeError(
                "QEC lowering dependencies need stable symbol names")
    attrs = {
        "sym_name":
            mlir_ir.StringAttr.get(symbol, context=context),
        "manifest_name":
            mlir_ir.StringAttr.get(definition.name, context=context),
        "manifest_sha256":
            mlir_ir.StringAttr.get(definition.manifest_sha256, context=context),
        "objective_family":
            mlir_ir.StringAttr.get(definition.objective_family,
                                   context=context),
        "codes":
            mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.FlatSymbolRefAttr.get(value, context=context)
                    for value in code_refs
                ],
                context=context,
            ),
        "requirements":
            mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(getattr(value, "key", str(value)),
                                           context=context)
                    for value in definition.requires
                ],
                context=context,
            ),
        "compiler_plugin":
            mlir_ir.StringAttr.get(definition.plugin, context=context),
        "compiler_symbol":
            mlir_ir.StringAttr.get(definition.compiler.name, context=context),
        "compiler_key":
            mlir_ir.StringAttr.get(
                getattr(
                    definition.compiler,
                    "key",
                    (f"{definition.plugin}:{definition.compiler.name}"
                     f"@{definition.version}"),
                ),
                context=context,
            ),
        "compiler_version":
            mlir_ir.StringAttr.get(definition.version, context=context),
        "dependencies":
            mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.FlatSymbolRefAttr.get(value, context=context)
                    for value in dependency_refs
                ],
                context=context,
            ),
        "input_stage":
            mlir_ir.StringAttr.get(definition.input_stage.value,
                                   context=context),
        "output_stage":
            mlir_ir.StringAttr.get(definition.output_stage.value,
                                   context=context),
        "provides_facets":
            mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(facet.value, context=context)
                    for facet in definition.provides_facets
                ],
                context=context,
            ),
    }
    objective = definition.objective
    if isinstance(objective, LogicalActionRef):
        attrs["objective"] = mlir_ir.Attribute.parse(
            f"#qlx.action<{objective.name}>", context=context)
    elif isinstance(objective, LogicalInstrumentRef):
        attrs["objective"] = mlir_ir.Attribute.parse(
            f"#qlx.instrument<{objective.name}>", context=context)
    elif isinstance(objective, ProgramDefinition):
        attrs["objective"] = mlir_ir.FlatSymbolRefAttr.get(
            transaction.materialize(objective).symbol, context=context)
    elif objective is not None:
        raise TypeError("QECLowering objective must be a typed logical action, "
                        "instrument, or @cudaq.logical.objective definition")
    policy = _dict(context, definition.policy_schema)
    metadata = _dict(context, definition.metadata)
    if policy is not None:
        attrs["policy_schema"] = policy
    if metadata is not None:
        attrs["metadata"] = metadata
    with transaction.location:
        operation = mlir_ir.Operation.create("qlx.qec_lowering",
                                             attributes=attrs,
                                             loc=transaction.location)
        transaction.module.body.append(operation)
    handle = DefinitionHandle(symbol, "qec_lowering", "common")
    transaction.bind(definition, handle)
    return handle

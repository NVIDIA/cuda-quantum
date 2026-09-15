# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.programs.definition import DefinitionHandle
from .build import Build, EvidenceRecord
from .context import CompilationContext


def _strings(context, values):
    return mlir_ir.ArrayAttr.get(
        [
            mlir_ir.StringAttr.get(str(value), context=context)
            for value in values
        ],
        context=context,
    )


def _plugin(context, value):
    if value is None:
        return None
    return mlir_ir.DictAttr.get(
        {
            key: mlir_ir.StringAttr.get(str(item), context=context)
            for key, item in value.items()
            if item is not None
        },
        context=context,
    )


def materialize_target(target, *, module=None) -> Build:
    """Materialize the portable manifest derived from an immutable Target."""

    manifest = target.manifest()
    transaction = CompilationContext(module=module)
    context = transaction.context
    location = transaction.location
    recipe_symbols = []
    with location:
        for capability, recipe in manifest["recipes"].items():
            symbol = transaction.unique_symbol(
                f"{manifest['name']}_{capability}_recipe")
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(symbol, context=context),
                "capability":
                    mlir_ir.StringAttr.get(capability, context=context),
                "accepted_stages":
                    _strings(context, recipe["accepted_stages"]),
                "required_facets":
                    _strings(context, recipe["required_facets"]),
                "provides_facets":
                    _strings(context, recipe["provides_facets"]),
                "stages":
                    _strings(context, recipe["stages"]),
                "finalizer":
                    mlir_ir.StringAttr.get(recipe["finalizer"],
                                           context=context),
                "effect":
                    mlir_ir.StringAttr.get(recipe["effect"], context=context),
            }
            if recipe["produced_stage"] is not None:
                attrs["produced_stage"] = mlir_ir.StringAttr.get(
                    recipe["produced_stage"], context=context)
            if recipe["result_schema"] is not None:
                attrs["result_schema"] = mlir_ir.StringAttr.get(
                    recipe["result_schema"], context=context)
            plugin = _plugin(context, manifest["plugin"])
            if plugin is not None:
                attrs["plugin"] = plugin
            transaction.module.body.append(
                mlir_ir.Operation.create("qlx.lowering_recipe",
                                         attributes=attrs,
                                         loc=location))
            recipe_symbols.append(symbol)

        target_symbol = transaction.unique_symbol(manifest["name"])
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(target_symbol, context=context),
            "capabilities":
                _strings(context, manifest["capabilities"]),
            "recipes":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.FlatSymbolRefAttr.get(value, context=context)
                        for value in recipe_symbols
                    ],
                    context=context,
                ),
            "availability":
                mlir_ir.StringAttr.get(manifest["availability"],
                                       context=context),
        }
        plugin = _plugin(context, manifest["plugin"])
        if plugin is not None:
            attrs["plugin"] = plugin
        transaction.module.body.append(
            mlir_ir.Operation.create("qlx.target_manifest",
                                     attributes=attrs,
                                     loc=location))

    profiles = {
        str(getattr(value, "value", value)).strip('"')
        for value in transaction.module.operation.attributes["qlx.profiles"]
    }
    profiles.add("common")
    transaction.module.operation.attributes["qlx.profiles"] = _strings(
        context, tuple(sorted(profiles)))
    if not transaction.module.operation.verify():
        raise ValueError("target manifest failed MLIR verification")
    portable = manifest["plugin"] is not None
    return Build(
        context=context,
        module=transaction.module,
        root=DefinitionHandle(target_symbol, "target_manifest", "common"),
        profile="common",
        pipeline=None,
        evidence=(EvidenceRecord(
            kind="target_manifest_verification",
            producer="cudaq-logical-python@0.3",
            result="pass" if portable else "unresolved",
            obligations=(
                "derived-capabilities",
                "ordered-recipes",
                *(() if portable else ("versioned-replay-provider",)),
            ),
        ),),
    )


__all__ = ["materialize_target"]

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.logical
import cudaq.mlir.ir as mlir_ir


def _objective(context: cudaq.logical.compiler.CompilationContext,
               name: str = "cached"):
    symbol = context.declare_objective(
        family="action",
        requested_symbol=name,
        kind="test",
        inputs=(),
        results=(),
    )
    return symbol, context.find_symbol(symbol)


def test_find_symbol_evicts_an_erased_cached_operation():
    context = cudaq.logical.compiler.CompilationContext()
    symbol, operation = _objective(context)

    operation.erase()

    assert context.find_symbol(symbol) is None
    assert context.unique_symbol(symbol) == symbol


def test_find_symbol_reindexes_a_renamed_cached_operation():
    context = cudaq.logical.compiler.CompilationContext()
    symbol, operation = _objective(context)
    operation.attributes["sym_name"] = mlir_ir.StringAttr.get(
        "renamed",
        context=context.context,
    )

    assert context.find_symbol(symbol) is None
    assert context.find_symbol("renamed") is operation
    assert context.find_symbol("renamed", "qlx.action") is operation
    assert context.find_symbol("renamed", "qlx.instrument_decl") is None


def test_find_symbol_can_skip_a_module_scan_when_the_index_is_complete(
        monkeypatch):
    context = cudaq.logical.compiler.CompilationContext()
    symbol, operation = _objective(context)

    def unexpected_walk():
        raise AssertionError("complete symbol index must not rescan MLIR")
        yield

    monkeypatch.setattr(context, "walk", unexpected_walk)

    assert context.find_symbol(symbol, "qlx.action", scan=False) is operation
    assert context.find_symbol("absent", scan=False) is None

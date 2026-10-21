# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.logical as cql
import cudaq.mlir.ir as mlir_ir
import pytest


def _objective(context: cql.compiler.CompilationContext, name: str = "cached"):
    symbol = context.declare_objective(
        family="action",
        requested_symbol=name,
        kind="test",
        inputs=(),
        results=(),
    )
    return symbol, context.find_symbol(symbol)


def test_find_symbol_evicts_an_erased_cached_operation():
    context = cql.compiler.CompilationContext()
    symbol, operation = _objective(context)

    operation.erase()

    assert context.find_symbol(symbol) is None
    assert context.unique_symbol(symbol) == symbol


def test_find_symbol_reindexes_a_renamed_cached_operation():
    context = cql.compiler.CompilationContext()
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
    context = cql.compiler.CompilationContext()
    symbol, operation = _objective(context)

    def unexpected_walk():
        raise AssertionError("complete symbol index must not rescan MLIR")
        yield

    monkeypatch.setattr(context, "walk", unexpected_walk)

    assert context.find_symbol(symbol, "qlx.action", scan=False) is operation
    assert context.find_symbol("absent", scan=False) is None


def test_unique_symbol_preserves_normalized_names_and_explicit_suffixes():
    context = cql.compiler.CompilationContext()
    requested = [
        "gate_2", "gate", "gate", "gate_1", "gate", "a b", "a_b", "9gate",
        "9gate"
    ]
    assert [context.unique_symbol(name) for name in requested] == [
        "gate_2", "gate", "gate_1", "gate_1_1", "gate_3", "a_b", "a_b_1",
        "_9gate", "_9gate_1"
    ]


@pytest.mark.parametrize("mutation", ["erase", "rename"])
@pytest.mark.parametrize("released_index", [0, 1, 3])
def test_unique_symbol_reuses_names_freed_after_suffix_allocation(
        mutation, released_index):
    context = cql.compiler.CompilationContext()
    declarations = [_objective(context) for _ in range(5)]
    symbol, operation = declarations[released_index]
    if mutation == "erase":
        operation.erase()
    else:
        operation.attributes["sym_name"] = mlir_ir.StringAttr.get(
            "renamed", context=context.context)

    assert context.find_symbol(symbol) is None
    assert context.unique_symbol("cached") == symbol
    assert context.unique_symbol("cached") == "cached_5"


def test_unique_symbol_skips_declarations_added_after_suffix_allocation():
    context = cql.compiler.CompilationContext()
    assert context.unique_symbol("cached") == "cached"
    assert context.unique_symbol("cached") == "cached_1"
    _objective(context, "cached_2")
    assert context.unique_symbol("cached") == "cached_3"


def test_unique_symbol_respects_symbols_in_a_replayed_module():
    original = cql.compiler.CompilationContext()
    _objective(original, "cached")
    _objective(original, "cached_2")
    context = cql.compiler.CompilationContext(module=original.module)
    assert context.unique_symbol("cached") == "cached_1"
    assert context.unique_symbol("cached") == "cached_3"

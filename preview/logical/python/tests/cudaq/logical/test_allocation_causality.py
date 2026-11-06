# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Allocation causality across narrow and wide physical call boundaries."""

import pytest

import cudaq.logical as cql
from cudaq.mlir import ir


def _resources(count):
    return "\n".join(
        f'phys.resource @q{i} {{kind = "qubit", resource_class = @q, '
        f'index = {i} : i64}}' for i in range(count))


def _types(count):
    return ", ".join(f"!phys.state<@q{i}>" for i in range(count))


def _references(count):
    return ", ".join(f"@q{i}" for i in range(count))


def _allocation(count):
    return (f'{{allocation = "anchors", resource_class = @q, '
            f'resources = [{_references(count)}], '
            f'indices = array<i64: {", ".join(map(str, range(count)))}>, '
            'acquire = "acquire", release = "release"}')


def _reuse_module(width, ordered):
    count = width + 1
    arguments = ", ".join(f"%a#{i}" for i in range(width))
    block_arguments = ", ".join(
        f"%arg{i}: !phys.state<@q{i}>" for i in range(width))
    yielded = ", ".join(f"%arg{i}" for i in range(width))
    second_index = 0 if ordered else width
    second_input = "%first#0" if ordered else f"%a#{width}"
    released = [f"%first#{i}" for i in range(width)] + [f"%a#{width}"]
    released[second_index] = "%second"
    return f"""
module attributes {{qlx.profiles = ["p3"]}} {{
  phys.machine @arch {{
    phys.resource_class @q {{kind = "qubit", count = {count + 1} : i64,
                            native_actions = []}}
  }}
  {_resources(count)}
  phys.resource @first {{kind = "qubit", resource_class = @q, index = {count} : i64}}
  phys.resource @second {{kind = "qubit", resource_class = @q, index = {count} : i64}}
  phys.graph @graph on @arch : () -> () {{
    %a:{count} = phys.acquire [{_references(count)}] {{event_id = "acquire"}}
      : {_types(count)}
    %first:{width} = "phys.call"({arguments})
      <{{callee = @first_user, instance = "first", event_id = "call0"}}> ({{
    ^bb0({block_arguments}):
      %scratch = phys.acquire [@first] {{event_id = "acquire0"}} : !phys.state<@first>
      phys.release %scratch {{event_id = "release0"}} : !phys.state<@first>
      phys.yield {yielded} : {_types(width)}
    }}) : ({_types(width)}) -> ({_types(width)})
    %bridge = "phys.call"({second_input})
      <{{callee = @bridge, instance = "bridge", event_id = "bridge"}}> ({{
    ^bb0(%arg: !phys.state<@q{second_index}>):
      phys.yield %arg : !phys.state<@q{second_index}>
    }}) : (!phys.state<@q{second_index}>) -> !phys.state<@q{second_index}>
    %second = "phys.call"(%bridge)
      <{{callee = @second_user, instance = "second", event_id = "call1"}}> ({{
    ^bb0(%arg: !phys.state<@q{second_index}>):
      %scratch = phys.acquire [@second] {{event_id = "acquire1"}} : !phys.state<@second>
      phys.release %scratch {{event_id = "release1"}} : !phys.state<@second>
      phys.yield %arg : !phys.state<@q{second_index}>
    }}) : (!phys.state<@q{second_index}>) -> !phys.state<@q{second_index}>
    phys.release {", ".join(released)} {{event_id = "release"}} : {_types(count)}
    phys.return
  }}
  phys.allocation_mapping @mapping for @graph {{entries = [
    {{allocation = "first", resource_class = @q, resources = [@first],
      indices = array<i64: {count}>, acquire = "acquire0", release = "release0"}},
    {{allocation = "second", resource_class = @q, resources = [@second],
      indices = array<i64: {count}>, acquire = "acquire1", release = "release1"}},
    {_allocation(count)}
  ]}}
}}
"""


@pytest.mark.parametrize("width", [2, 9])
@pytest.mark.parametrize("ordered", [False, True])
def test_shared_release_does_not_order_independent_lifetimes(width, ordered):
    context = cql.compiler.CompilationContext().context
    source = _reuse_module(width, ordered)
    if ordered:
        assert ir.Module.parse(source, context=context).operation.verify()
    else:
        with pytest.raises(ir.MLIRError, match="causally unordered adjacent"):
            ir.Module.parse(source, context=context)


@pytest.mark.parametrize("width", [2, 9])
def test_acquire_reaches_release_inside_a_direct_user(width):
    context = cql.compiler.CompilationContext().context
    inputs = ", ".join(f"%a#{i}" for i in range(width))
    block_arguments = ", ".join(
        f"%arg{i}: !phys.state<@q{i}>" for i in range(width))
    released = ", ".join(f"%arg{i}" for i in range(width))
    source = f"""
module attributes {{qlx.profiles = ["p3"]}} {{
  phys.machine @arch {{
    phys.resource_class @q {{kind = "qubit", count = {width} : i64,
                            native_actions = []}}
  }}
  {_resources(width)}
  phys.graph @graph on @arch : () -> () {{
    "phys.call"() <{{callee = @outer, instance = "outer", event_id = "outer"}}> ({{
      %a:{width} = phys.acquire [{_references(width)}] {{event_id = "acquire"}}
        : {_types(width)}
      "phys.call"({inputs})
        <{{callee = @inner, instance = "inner", event_id = "inner"}}> ({{
      ^bb0({block_arguments}):
        phys.release {released} {{event_id = "release"}} : {_types(width)}
        phys.yield
      }}) : ({_types(width)}) -> ()
      phys.yield
    }}) : () -> ()
    phys.return
  }}
  phys.allocation_mapping @mapping for @graph {{entries = [{_allocation(width)}]}}
}}
"""
    assert ir.Module.parse(source, context=context).operation.verify()

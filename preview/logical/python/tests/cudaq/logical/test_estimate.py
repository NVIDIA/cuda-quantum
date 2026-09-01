# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical and static estimate coverage retained by the product preview."""

import json

import pytest

import cudaq.logical
from cudaq.logical.analysis import Tier


@cudaq.logical.code
class TinyCode:
    block = cudaq.logical.codes.CSSBlock(data=3, sx=0, sz=2)
    d = 3
    hx = ()
    hz = ((0, 1), (1, 2))
    lx = ((0, 1, 2),)
    lz = ((0,),)


@cudaq.logical.gadget(implements=cudaq.logical.std.h)
def encoded_h(
        block: cudaq.logical.patch[TinyCode]) -> cudaq.logical.patch[TinyCode]:
    return cudaq.logical.h(block.data)


@cudaq.logical.protocol(implements=cudaq.logical.std.idle)
def four_h(
        block: cudaq.logical.patch[TinyCode]) -> cudaq.logical.patch[TinyCode]:
    return cudaq.logical.ops.repeat(4,
                                    carries=(block,),
                                    body=lambda value: encoded_h(value))


def test_callable_estimate_namespace_preserves_folded_logical_counts():

    @cudaq.logical.program
    def folded() -> bool:
        q = cudaq.logical.prepare_zero()
        q, = cudaq.logical.ops.repeat(5,
                                      carries=(q,),
                                      body=lambda _i, value:
                                      (cudaq.logical.h(value),))
        return cudaq.logical.measure_z(q)

    build = cudaq.logical.compile(folded)
    profile = cudaq.logical.estimate(build, tier=Tier.LOGICAL)

    assert profile.actions["qlx_standard_h"] == 5
    assert profile.instruments["qlx_standard_prepare_zero"] == 1
    assert profile.instruments["qlx_standard_measure_z"] == 1
    assert profile.logical_qubits_peak == 1
    assert profile.action_depth_upper_bound == 7
    assert profile.build_root == build.root.symbol
    assert profile.build_sha256 == build.content_sha256


def test_in_scope_estimate_results_project_to_plain_data():

    @cudaq.logical.program
    def portable() -> bool:
        return cudaq.logical.measure_z(cudaq.logical.prepare_zero())

    logical = cudaq.logical.estimate(portable, tier=Tier.LOGICAL)
    static = cudaq.logical.estimate(cudaq.logical.compile(four_h))

    logical_payload = {
        "actions": dict(logical.actions),
        "instruments": dict(logical.instruments),
    }
    static_payload = {
        "operations": dict(static.operation_counts),
        "gadgets": dict(static.gadget_calls),
    }
    for payload in (logical_payload, static_payload):
        assert json.loads(json.dumps(payload)) == payload


def test_estimate_results_rehydrate_from_cudaq_annotations():

    @cudaq.logical.program
    def portable() -> bool:
        return cudaq.logical.measure_z(cudaq.logical.prepare_zero())

    logical = cudaq.logical.estimate(portable, tier=Tier.LOGICAL)
    static = cudaq.logical.estimate(cudaq.logical.compile(four_h))
    annotations = {
        Tier.LOGICAL.name: logical.to_dict(),
        Tier.STATIC.name: static.to_dict(),
    }

    assert cudaq.logical.estimate.LogicalProfile.from_annotations(
        annotations) == logical
    assert cudaq.logical.estimate.FabricCounts.from_annotations(
        annotations) == static


def test_static_estimate_expands_folded_call_multiplicity_analytically():
    build = cudaq.logical.compile(four_h)
    counts = cudaq.logical.estimate(build)

    assert counts.source_stage == "p2"
    assert "protocol_network" in counts.source_facets
    assert counts.operation_counts["repeat"] == 1
    assert counts.operation_counts["h"] == 4
    assert counts.gadget_calls["encoded_h"] == 4
    assert counts.logical_qubits_peak == 1
    assert counts.total_operations == 4
    assert counts.build_root == build.root.symbol
    assert counts.build_sha256 == build.content_sha256


def test_static_estimate_rejects_a_program_without_selected_p2_evidence():

    @cudaq.logical.program
    def portable() -> bool:
        return cudaq.logical.measure_z(cudaq.logical.prepare_zero())

    with pytest.raises(ValueError, match="selected P2"):
        cudaq.logical.estimate(cudaq.logical.compile(portable))

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Patch adjacency follows `unowned` carriers without crossing other patches."""

from itertools import combinations
import random

import pytest

from cudaq.logical.architecture import PatchKind, PatchTopology, Topology


def _bind(groups, topology, *, categories=None):
    return PatchTopology(groups, categories=categories)._bind(
        topology=topology,
        capacity=len(groups),
        resource_count=len(topology.nodes),
    )


@pytest.mark.parametrize(
    "groups, expected",
    [
        (((0,),), ()),
        (((0,), (4,)), ((0, 1),)),
        (((0,), (2,), (4,)), ((0, 1), (1, 2))),
        # Every source carrier must seed the search, even disconnected ones.
        (((0, 4), (1,), (3,)), ((0, 1), (0, 2), (1, 2))),
        # Slot zero must also block traversal when it is not the source.
        (((2,), (0,), (4,)), ((0, 1), (0, 2))),
    ],
)
def test_patch_paths_do_not_cross_owned_carriers(groups, expected):
    categories = {0: PatchKind.DATA}
    result = _bind(groups, Topology("line", length=5), categories=categories)
    assert result.edges == expected
    assert result.carrier_groups == groups
    assert result.categories == (PatchKind.DATA,) + (None,) * (len(groups) - 1)


def _expected_edges(groups, topology):
    # Independent pairwise oracle: remove every third patch, then ask the
    # carrier topology whether either endpoint group can reach the other.
    unowned = set(topology.nodes).difference(*groups)
    return tuple(
        (left, right)
        for left, right in combinations(range(len(groups)), 2)
        if any(
            topology.shortest_path(source,
                                   target,
                                   allowed=unowned.union(
                                       groups[left], groups[right])) is not None
            for source in groups[left]
            for target in groups[right]))


def test_patch_adjacency_matches_all_four_carrier_graphs():
    possible_edges = tuple(combinations(range(4), 2))
    for mask in range(1 << len(possible_edges)):
        topology = Topology(
            num_nodes=4,
            edges=tuple(edge for bit, edge in enumerate(possible_edges)
                        if mask & (1 << bit)),
        )
        for groups in (((0,), (1,), (2,), (3,)), ((0,), (3,)),
                       ((0, 2), (1,), (3,)), ((2,), (0,), (3,))):
            assert _bind(groups,
                         topology).edges == _expected_edges(groups, topology)


def test_patch_adjacency_matches_random_graphs():
    rng = random.Random(2026)
    for _ in range(100):
        nodes = list(range(24))
        rng.shuffle(nodes)
        groups = tuple(
            tuple(nodes[start:start + 2]) for start in range(0, 12, 2))
        topology = Topology(
            num_nodes=24,
            edges=tuple(
                edge for edge in combinations(nodes, 2) if rng.random() < 0.08),
        )
        assert _bind(groups,
                     topology).edges == _expected_edges(groups, topology)


def test_qec_physical_binding_derives_patch_adjacency():
    from cudaq.logical import codes
    from cudaq.logical.architecture import ResourceClass
    from cudaq.logical.devices import QECRegion, QECToPhysicalBinding

    region = QECRegion("compute",
                       codes.Encoding(codes.Surface[3]),
                       block_capacity=3)
    patches = PatchTopology(tuple(
        tuple(range(10 * slot, 10 * slot + 9)) for slot in range(3)),
                            categories={1: PatchKind.ANCILLA})
    binding = QECToPhysicalBinding(
        region,
        (ResourceClass("qubit", 29),),
        topology=Topology("line", length=29),
        patch_topology=patches,
    )
    assert binding.patch_topology.edges == ((0, 1), (1, 2))
    assert binding.patch_topology.carrier_groups == patches.carrier_groups
    assert binding.patch_topology.categories == patches.categories
    assert patches.edges == ()  # Binding must preserve the authored value.

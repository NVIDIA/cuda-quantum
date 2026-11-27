# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""First-fit placement with sparse reservations and many allocations."""

import pytest

from cudaq.logical.architecture.logical import LogicalMachine, Space
from cudaq.logical.compiler.placement_allocator import _Allocator


@pytest.mark.parametrize("capacity", [8, None])
def test_first_fit_preserves_holes_around_explicit_reservations(capacity):
    machine = LogicalMachine(
        "machine", {"compute": Space(name="compute", capacity=capacity)})
    space = machine.spaces[0]
    allocator = _Allocator(machine, ())
    assert allocator.take(space=space, slot=5) == (space, 5)
    assert allocator.take() == (space, 0)
    assert allocator.take(space=space, slot=2) == (space, 2)
    with pytest.raises(ValueError, match="already occupied"):
        allocator.take(space=space, slot=0)
    assert [allocator.take()[1] for _ in range(5)] == [1, 3, 4, 6, 7]
    if capacity is not None:
        with pytest.raises(ValueError, match="capacity is exhausted"):
            allocator.take()
    else:
        assert allocator.take() == (space, 8)


def test_capacity_checks_and_cursors_are_independent_per_space():
    machine = LogicalMachine(
        "machine", {
            "first": Space(name="first", capacity=3),
            "second": Space(name="second", capacity=2),
        })
    first, second = machine.spaces
    allocator = _Allocator(machine, ())
    assert allocator.take(space=first, slot=1) == (first, 1)
    with pytest.raises(ValueError, match="capacity is exhausted"):
        allocator.take(minimum_remaining=3)
    assert allocator.take(minimum_remaining=2) == (first, 0)
    assert allocator.take(minimum_remaining=2) == (second, 0)
    assert allocator.take() == (first, 2)
    assert allocator.take() == (second, 1)
    with pytest.raises(ValueError, match="capacity is exhausted"):
        allocator.take()


@pytest.mark.parametrize("capacity", [1024, None])
def test_first_fit_does_not_rescan_the_occupied_prefix(capacity):

    class CountedSlots(set):
        probes = 0

        def __contains__(self, slot):
            self.probes += 1
            return super().__contains__(slot)

    machine = LogicalMachine(
        "machine", {"compute": Space(name="compute", capacity=capacity)})
    space = machine.spaces[0]
    allocator = _Allocator(machine, ())
    slots = CountedSlots()
    allocator.used_slots[space.name] = slots
    assert [allocator.take()[1] for _ in range(1024)] == list(range(1024))
    assert slots.probes <= 2 * 1024


@pytest.mark.parametrize("capacity", [0, 2])
def test_full_space_is_skipped_even_without_a_remaining_demand(capacity):
    machine = LogicalMachine(
        "machine", {"compute": Space(name="compute", capacity=capacity)})
    space = machine.spaces[0]
    allocator = _Allocator(machine, ())
    for expected in range(capacity):
        assert allocator.take(minimum_remaining=0) == (space, expected)
    with pytest.raises(ValueError, match="capacity is exhausted"):
        allocator.take(minimum_remaining=0)

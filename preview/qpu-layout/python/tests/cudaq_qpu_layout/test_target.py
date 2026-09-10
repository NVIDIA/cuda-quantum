# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The target end of things: `cudaq.set_target` through to a trace."""

import cudaq
import pytest

from cudaq_qpu_layout import QpuLayoutTarget


@pytest.fixture(autouse=True)
def reset():
    yield
    cudaq.reset_target()


@cudaq.kernel
def bell_cross():
    """Two Bell pairs plus one cross-pair gate -- the running example."""
    q = cudaq.qvector(4)
    h(q[0])
    x.ctrl(q[0], q[1])
    h(q[2])
    x.ctrl(q[2], q[3])
    x.ctrl(q[1], q[2])


@cudaq.kernel
def rolled():
    """A loop, to prove the endpoint unrolls what `nop` codegen leaves rolled."""
    q = cudaq.qvector(3)
    for i in range(3):
        h(q[i])


def run(kernel, **model):
    target = QpuLayoutTarget.build(**model)
    cudaq.set_target(target)
    cudaq.sample(kernel, shots_count=10)
    return target.runtime_endpoint.trace


def test_two_regions_force_one_crossing():
    """The cross-pair gate is the only thing that cannot stay put."""
    trace = run(bell_cross, num_regions=2, region_size=2)
    assert trace["summary"]["num_vqubits"] == 4
    assert trace["summary"]["moves"]["cross"] == 2
    assert trace["summary"]["total_move_cost"] > 0


def test_one_big_region_needs_no_movement():
    trace = run(bell_cross, num_regions=1, region_size=4)
    assert trace["summary"]["total_move_cost"] == 0
    assert not [m for s in trace["steps"] for m in s["moves"]]


def test_bell_pairs_run_in_parallel():
    """Disjoint pairs share timesteps, which is the point of the trace."""
    trace = run(bell_cross, num_regions=1, region_size=4)
    assert trace["summary"]["depth"] < 5


def test_loops_are_unrolled_by_the_endpoint():
    trace = run(rolled, num_regions=1, region_size=3)
    gates = [o for s in trace["steps"] for o in s["ops"]]
    assert len(gates) == 3
    assert {g["gate"] for g in gates} == {"h"}


def test_model_configuration_reaches_the_simulator():
    trace = run(bell_cross, num_regions=2, region_size=2, move_cost=7)
    assert trace["model"]["move_cost"] == 7
    crossings = [m for s in trace["steps"] for m in s["moves"]
                 if m["kind"] == "cross"]
    assert crossings and all(m["cost"] == 7 for m in crossings)


def test_counts_are_zero_and_shaped_like_the_kernel():
    """Nothing simulates a state, so every shot reads back zero."""
    target = QpuLayoutTarget.build(num_regions=2, region_size=2)
    cudaq.set_target(target)
    counts = cudaq.sample(bell_cross, shots_count=10)
    assert dict(counts.items()) == {"0000": 10}

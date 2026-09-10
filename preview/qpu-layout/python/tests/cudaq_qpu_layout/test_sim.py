# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Invariants the layout simulator must uphold, plus a few worked examples."""

import os
import pytest

from cudaq_qpu_layout.model import QpuModel
from cudaq_qpu_layout.sim import simulate, LayoutError
from cudaq_qpu_layout.trace import (replay, COMPUTE, IN, OUT, CROSS,
                                    PORT_IN, PORT_OUT)

PAYLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "payloads")


def run(name, **kwargs):
    with open(os.path.join(PAYLOADS, name)) as f:
        return simulate(f.read(), QpuModel(**kwargs)).to_json()


# ===----------------------------------------------------------------------=== #
# Invariants
# ===----------------------------------------------------------------------=== #


@pytest.fixture(params=[
    ("bell_cross.mlir", dict(num_regions=2, region_size=2)),
    ("bell_cross.mlir", dict(num_regions=4, region_size=2)),
    ("bell_cross.mlir", dict(num_regions=1, region_size=4)),
    ("line_chain.mlir", dict(num_regions=1, region_size=4)),
    ("line_chain.mlir", dict(num_regions=2, region_size=3)),
])
def trace(request):
    name, model = request.param
    return run(name, **model)


def test_operands_are_colocated(trace):
    """A gate can only run on qubits sharing one region."""
    for step in trace["steps"]:
        for op in step["ops"]:
            regions = {q["region"] for q in op["controls"] + op["targets"]}
            assert regions == {op["region"]}, \
                f"op {op['gate']} at t={step['t']} spans regions {regions}"


def test_capacity_is_never_exceeded(trace):
    """Compute wires are the scarce resource; ports are unbounded staging."""
    size = trace["model"]["region_size"]
    for t, pos in replay(trace):
        counts = {}
        for region, kind, _ in pos.values():
            if kind == COMPUTE:
                counts[region] = counts.get(region, 0) + 1
        for region, n in counts.items():
            assert n <= size, f"region {region} holds {n} > {size} at t={t}"


def test_ops_only_run_on_compute_wires(trace):
    for (t, pos), step in zip(replay(trace), trace["steps"]):
        for op in step["ops"]:
            for q in op["controls"] + op["targets"]:
                assert pos[q["vq"]][1] == COMPUTE, \
                    f"op {op['gate']} at t={t} uses vq {q['vq']} on a port"


def test_every_crossing_is_bracketed_by_ports(trace):
    """A region is entered and left only through ports, so each qubit's move
    sequence must read port-out, cross, port-in."""
    per_qubit = {}
    for step in trace["steps"]:
        for m in step["moves"]:
            per_qubit.setdefault(m["vq"], []).append(m)
    for vq, moves in per_qubit.items():
        for i, m in enumerate(moves):
            if m["kind"] != CROSS:
                continue
            assert m["from"]["kind"] == OUT and m["to"]["kind"] == IN, m
            assert i > 0 and moves[i - 1]["kind"] == PORT_OUT, \
                f"vq {vq} crossed without first reaching an out-port"
            assert i + 1 < len(moves) and moves[i + 1]["kind"] == PORT_IN, \
                f"vq {vq} crossed but never landed on a compute wire"


def test_port_hops_come_in_pairs(trace):
    summary = trace["summary"]["moves"]
    assert summary["port"] == 2 * summary["cross"], summary


def test_no_qubit_is_released_from_a_port(trace):
    """A port is transit, not storage: a qubit ends its life on a compute wire."""
    for step in trace["steps"]:
        for d in step["deltas"]:
            if d["event"] == "release":
                assert d["from"]["kind"] == COMPUTE, \
                    f"vq {d['vq']} was released while parked on a port"


def test_no_two_qubits_share_a_slot(trace):
    """The delta stream must replay to a consistent occupancy."""
    for t, pos in replay(trace):
        sites = list(pos.values())
        assert len(sites) == len(set(sites)), \
            f"two qubits occupy the same slot at t={t}: {pos}"


def test_ops_reference_the_replayed_placement(trace):
    """Each op's recorded slots agree with the state the deltas reconstruct."""
    for (t, pos), step in zip(replay(trace), trace["steps"]):
        for op in step["ops"]:
            for q in op["controls"] + op["targets"]:
                assert pos[q["vq"]] == (q["region"], COMPUTE, q["slot"]), \
                    f"op {op['gate']} at t={t} disagrees on vq {q['vq']}"


def test_summary_matches_steps(trace):
    moves = [m for s in trace["steps"] for m in s["moves"]]
    summary = trace["summary"]
    assert summary["moves"]["cross"] == sum(1 for m in moves
                                            if m["kind"] == CROSS)
    assert summary["moves"]["port"] == sum(
        1 for m in moves if m["kind"] in (PORT_IN, PORT_OUT))
    assert summary["total_move_cost"] == sum(m["cost"] for m in moves)
    op_steps = [s["t"] for s in trace["steps"] if s["ops"]]
    assert summary["depth"] == (max(op_steps) + 1 if op_steps else 0)


def test_moves_are_between_distinct_sites(trace):
    for step in trace["steps"]:
        for m in step["moves"]:
            assert m["from"] != m["to"]
            src, dst = m["from"], m["to"]
            if m["kind"] == PORT_OUT:
                assert src["region"] == dst["region"]
                assert (src["kind"], dst["kind"]) == (COMPUTE, OUT)
            elif m["kind"] == PORT_IN:
                assert src["region"] == dst["region"]
                assert (src["kind"], dst["kind"]) == (IN, COMPUTE)
            else:
                assert m["kind"] == CROSS
                assert (src["kind"], dst["kind"]) == (OUT, IN)


# ===----------------------------------------------------------------------=== #
# Worked examples
# ===----------------------------------------------------------------------=== #


def test_independent_pairs_run_in_parallel():
    """Two Bell pairs on two regions occupy two timesteps, not four."""
    trace = run("bell_cross.mlir", num_regions=2, region_size=2)
    assert [len(s["ops"]) for s in trace["steps"][:2]] == [2, 2]
    assert {op["region"] for op in trace["steps"][0]["ops"]} == {0, 1}


def test_cross_pair_displaces_a_resident():
    """With 4 live qubits on 4 compute wires, bringing the two cross-pair qubits
    together has no free wire to land on: a resident must vacate through an
    out-port and be carried to the region the incoming qubit left. That is two
    crossings, each bracketed by a port hop."""
    trace = run("bell_cross.mlir", num_regions=2, region_size=2)
    assert trace["summary"]["moves"] == {"cross": 2, "port": 4}
    # move_cost 4 twice, port_cost 1 four times.
    assert trace["summary"]["total_move_cost"] == 12

    # The displaced qubit ends up on a compute wire in the other region.
    final = list(replay(trace))[-1][1]
    assert all(kind == COMPUTE for _, kind, _ in final.values()), final


def test_uncontended_transfer_is_a_single_crossing():
    """Given a free compute wire to land on, no resident is displaced.

    vq0 and vq1 fill region 0, so vq2 lands in region 1. Releasing vq1 frees a
    wire in region 0, so the final gate draws vq2 in with one crossing.
    """
    src = """
    quake.wire_set @wires[3]
    func.func @handoff() attributes {"cudaq-entrypoint"} {
      %q0 = quake.borrow_wire @wires[0] : !quake.wire
      %q1 = quake.borrow_wire @wires[1] : !quake.wire
      %q2 = quake.borrow_wire @wires[2] : !quake.wire
      %a = quake.h %q0 : (!quake.wire) -> !quake.wire
      %b = quake.h %q1 : (!quake.wire) -> !quake.wire
      %c = quake.h %q2 : (!quake.wire) -> !quake.wire
      quake.return_wire %b : !quake.wire
      %d:2 = quake.x [%a] %c
          : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      quake.return_wire %d#0 : !quake.wire
      quake.return_wire %d#1 : !quake.wire
      return
    }
    """
    trace = simulate(src, QpuModel(num_regions=2, region_size=2)).to_json()
    assert trace["summary"]["moves"] == {"cross": 1, "port": 2}


def test_one_big_region_needs_no_movement():
    """With room for everything in one region, nothing ever moves."""
    trace = run("bell_cross.mlir", num_regions=1, region_size=4)
    assert trace["summary"]["total_move_cost"] == 0


def test_co_located_qubits_never_move():
    """A region is all-to-all, so a gate on two residents costs no movement."""
    trace = run("line_chain.mlir", num_regions=1, region_size=4)
    assert not [m for s in trace["steps"] for m in s["moves"]]
    assert trace["summary"]["total_move_cost"] == 0


def test_measurements_are_counted():
    trace = run("line_chain.mlir", num_regions=1, region_size=4)
    assert trace["summary"]["num_measurements"] == 1


def test_control_flow_is_rejected():
    with pytest.raises(LayoutError, match="straight-line"):
        run("has_loop.mlir")


def test_over_capacity_is_reported():
    with pytest.raises(LayoutError, match="no free slot"):
        run("bell_cross.mlir", num_regions=1, region_size=2)


def test_multi_qubit_gate_is_supported():
    """An all-to-all region places no arity limit on an operation."""
    src = """
    quake.wire_set @wires[3]
    func.func @toffoli() attributes {"cudaq-entrypoint"} {
      %q0 = quake.borrow_wire @wires[0] : !quake.wire
      %q1 = quake.borrow_wire @wires[1] : !quake.wire
      %q2 = quake.borrow_wire @wires[2] : !quake.wire
      %r:3 = quake.x [%q0, %q1] %q2
          : (!quake.wire, !quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire, !quake.wire)
      quake.return_wire %r#0 : !quake.wire
      quake.return_wire %r#1 : !quake.wire
      quake.return_wire %r#2 : !quake.wire
      return
    }
    """
    builder = simulate(src, QpuModel(num_regions=1, region_size=3))
    assert builder.to_json()["summary"]["total_move_cost"] == 0

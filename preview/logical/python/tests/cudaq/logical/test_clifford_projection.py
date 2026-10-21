# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import cudaq.logical as cql
import cudaq.mlir.ir as ir
from cudaq.mlir._mlir_libs import _qlxRuntime


@cql.gadget(implements=cql.logical.h)
def bare_h(
        block: cql.patch[cql.codes.BareQubit]
) -> cql.patch[cql.codes.BareQubit]:
    return cql.h(block.data)


@cql.gadget(implements=cql.logical.idle)
def recorded_measurement(
    block: cql.patch[cql.codes.BareQubit],) -> cql.patch[cql.codes.BareQubit]:
    block, _ = cql.ops.measure_pauli(block.data, paulis="Z", record="read")
    return block


@cql.gadget(implements=cql.logical.idle)
def two_calls(
        block: cql.patch[cql.codes.BareQubit]
) -> cql.patch[cql.codes.BareQubit]:
    return recorded_measurement(recorded_measurement(block))


@cql.gadget(implements=cql.logical.idle)
def steane_round(
        block: cql.patch[cql.codes.Steane]) -> cql.patch[cql.codes.Steane]:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


@cql.gadget(implements=cql.logical.measure_z)
def bare_readout(block: cql.patch[cql.codes.BareQubit]) -> bool:
    block, result = cql.ops.mpp(cql.algebra.Z(block[0]))
    cql.discard(block)
    return result


def test_projection_is_typed_repeatable_and_preserves_the_build():
    build = cql.compile(bare_h)
    before = build.to_mlir(), build.content_sha256
    result = cql.compiler.project_clifford(build)
    assert result == cql.compiler.project_clifford(build)
    assert (build.to_mlir(), build.content_sha256) == before
    assert result.instructions == (cql.compiler.CliffordInstruction(
        "H", (0,), (), False, None, "op0.0"),)
    assert result.interface.carrier_count == 1
    assert result.interface.inputs[0].port_ordinal == 0
    assert result.interface.inputs[0].data_carriers == (0,)
    assert result.interface.outputs[0].data_carriers == (0,)
    assert result.interface.inputs[0].patch_id == result.interface.outputs[
        0].patch_id
    assert result.interface.measurements == ()
    with pytest.raises(FrozenInstanceError):
        result.instructions[0].site_id = "changed"
    with pytest.raises(TypeError):
        result.record_indices["changed"] = 0


def test_call_instances_have_unique_sites_and_measurement_paths():
    build = cql.compile(two_calls)
    projection = cql.compiler.project_clifford(build)
    assert [op.kind for op in projection.instructions] == ["MPP"] * 2
    assert [op.measurement_start for op in projection.instructions] == [0, 1]
    sites = [op.site_id for op in projection.instructions]
    records = [row.record for row in projection.interface.measurements]
    assert len(set(sites)) == len(set(records)) == 2
    assert all("__qlx_call" in site for site in sites)
    assert all(
        row.carriers == (0,) for row in projection.interface.measurements)
    assert all(row.symbol == "recorded_measurement"
               for row in projection.interface.measurements)
    assert [
        record.name for record in projection.interface.record_order(two_calls)
    ] == records
    assert all(projection.record_indices[f"two_calls.{record}"] == index
               for index, record in enumerate(records))

    selected = cql.compiler.project_clifford(build,
                                             gadget="recorded_measurement")
    assert len(selected.instructions) == 1
    assert selected.interface.measurements[0].record == "read.outcome"
    assert selected.interface.measurements[0].call_path == ()


def test_projection_uses_the_authenticated_build_snapshot():
    build = cql.compile(bare_h)
    expected = cql.compiler.project_clifford(build)
    inspection = build.module
    gadget = next(op for op in inspection.body.operations
                  if op.operation.name == "fabric.gadget")
    operation = next(op for op in gadget.regions[0].blocks[0].operations
                     if op.operation.name == "fabric.h")
    operation.attributes["indices"] = ir.DenseI64ArrayAttr.get(
        [999], context=inspection.context)
    assert cql.compiler.project_clifford(build) == expected


def test_cross_patch_gate_operands_follow_the_encoded_port_order():

    @cql.gadget(implements=cql.logical.cx)
    def bare_cx(
        control: cql.patch[cql.codes.BareQubit],
        target: cql.patch[cql.codes.BareQubit],
    ) -> tuple[cql.patch[cql.codes.BareQubit], cql.patch[cql.codes.BareQubit]]:
        return cql.cx(control.data, target.data)

    projection = cql.compiler.project_clifford(cql.compile(bare_cx))
    gate, = projection.instructions
    assert (gate.kind, gate.carriers) == ("CX", (0, 1))
    assert [port.port_ordinal for port in projection.interface.inputs] == [0, 1]
    assert [port.data_carriers for port in projection.interface.inputs
           ] == [(0,), (1,)]
    assert [port.data_carriers for port in projection.interface.outputs
           ] == [(0,), (1,)]
    assert len({port.patch_id for port in projection.interface.inputs}) == 2


def test_syndrome_assembly_aliases_reuse_physical_measurement_indices():
    projection = cql.compiler.project_clifford(cql.compile(steane_round))
    assert len(projection.interface.measurements) == 6
    assert projection.interface.inputs[0].carriers("data") == tuple(range(7))
    assert projection.interface.inputs[0].carriers("sx") == (7, 8, 9)
    assert projection.interface.inputs[0].carriers("sz") == (10, 11, 12)
    for index in range(6):
        assert projection.record_indices[f"round.s{index}"] == index
        assert projection.record_indices[
            f"steane_round.round.s{index}"] == index
    assert [row.carriers for row in projection.interface.measurements
           ] == [(carrier,) for carrier in range(7, 13)]


def test_nested_repeats_expand_distinct_invocation_paths_and_reject_dynamic_control(
):
    text = (Path(__file__).resolve().parents[4] /
            "test/Dialect/Fabric/clifford-projection.mlir").read_text()
    with ir.Context() as context:
        module = ir.Module.parse(text, context=context)
        before = str(module)
        result = json.loads(
            _qlxRuntime.project_clifford_json(module, "repeated"))
        assert str(module) == before
        assert [op["measurement_start"] for op in result["instructions"]
               ] == list(range(4))
        sites = [op["site_id"] for op in result["instructions"]]
        assert len(set(sites)) == 4
        assert sites[0] == "__qlx_repeat0_0.__qlx_repeat0_0.__qlx_call0.op0.0"
        assert sites[-1] == "__qlx_repeat0_1.__qlx_repeat0_1.__qlx_call0.op0.0"
        assert [row["index"] for row in result["measurements"]
               ] == list(range(4))
        assert len({row["record"] for row in result["measurements"]}) == 4
        with pytest.raises(RuntimeError, match="recursive executable call"):
            _qlxRuntime.project_clifford_json(module, "recursive")
        with pytest.raises(RuntimeError, match="conditional control flow"):
            _qlxRuntime.project_clifford_json(module, "dynamic")


def test_preparation_and_destructive_readout_have_different_boundaries():
    prepare = cql.gadgets.prepare_zero(cql.codes.BareQubit)
    preparation = cql.compiler.project_clifford(cql.compile(prepare))
    assert preparation.interface.inputs == ()
    assert preparation.interface.output_data_carriers == (0,)
    assert [op.kind for op in preparation.instructions] == ["R"]

    readout = cql.compiler.project_clifford(cql.compile(bare_readout))
    assert readout.interface.input_data_carriers == (0,)
    assert readout.interface.outputs == ()
    assert readout.instructions[0].kind == "MPP"
    assert readout.instructions[0].paulis == ((0, "Z"),)
    assert readout.instructions[0].measurement_start == 0
    assert readout.interface.measurements[0].record == "mpp0.outcome"


def test_projection_keeps_signed_physical_pauli_factors():

    @cql.gadget(implements=cql.logical.idle)
    def signed_product(
        block: cql.patch[cql.codes.BareQubit],
    ) -> cql.patch[cql.codes.BareQubit]:
        block, _ = cql.ops.measure_pauli(block.data,
                                         paulis="-Y",
                                         record="negative_y")
        return block

    projection = cql.compiler.project_clifford(cql.compile(signed_product))
    measurement, = projection.instructions
    assert (measurement.kind, measurement.paulis,
            measurement.inverted) == ("MPP", ((0, "Y"),), True)
    assert projection.interface.measurements[0].record == "negative_y.outcome"


def test_projection_rejects_nonclifford_and_unresolved_gadgets():

    @cql.gadget(implements=cql.logical.t)
    def nonclifford(
        block: cql.patch[cql.codes.BareQubit],
    ) -> cql.patch[cql.codes.BareQubit]:
        return cql.t(block.data)

    with pytest.raises(RuntimeError,
                       match="non-Clifford|unsupported operation"):
        cql.compiler.project_clifford(cql.compile(nonclifford))
    build = cql.compile(bare_h)
    with pytest.raises(RuntimeError,
                       match="selected gadget @missing does not resolve"):
        cql.compiler.project_clifford(build, gadget="missing")


def test_projection_requires_linked_spec_and_stable_measurement_names():
    build = cql.compile(recorded_measurement)
    module = _qlxRuntime.clone_module(build.module)
    gadget = next(op for op in module.body.operations
                  if op.operation.name == "fabric.gadget")
    del gadget.attributes["spec"]
    with pytest.raises(RuntimeError, match="requires a linked GadgetSpec"):
        _qlxRuntime.project_clifford_json(module, "recorded_measurement")

    module = _qlxRuntime.clone_module(build.module)
    gadget = next(op for op in module.body.operations
                  if op.operation.name == "fabric.gadget")
    measurement = next(op for op in gadget.regions[0].blocks[0].operations
                       if op.operation.name == "fabric.mpp")
    del measurement.attributes["record"]
    with pytest.raises(RuntimeError, match="stable record|record_schema"):
        _qlxRuntime.project_clifford_json(module, "recorded_measurement")


def test_projection_rejects_reserved_record_names_without_a_record_schema():
    build = cql.compile(recorded_measurement)
    module = _qlxRuntime.clone_module(build.module)
    spec = next(op for op in module.body.operations
                if op.operation.name == "fabric.gadget_spec")
    if "record_schema" in spec.attributes:
        del spec.attributes["record_schema"]
    gadget = next(op for op in module.body.operations
                  if op.operation.name == "fabric.gadget")
    measurement = next(op for op in gadget.regions[0].blocks[0].operations
                       if op.operation.name == "fabric.mpp")
    measurement.attributes["record"] = ir.StringAttr.get("__qlx_reserved",
                                                         context=module.context)
    with pytest.raises(RuntimeError, match="reserved"):
        _qlxRuntime.project_clifford_json(module, "recorded_measurement")


def test_projection_rejects_ambiguous_relative_and_qualified_aliases():

    @cql.gadget(implements=cql.logical.idle)
    def ambiguous_alias(
        block: cql.patch[cql.codes.BareQubit],
    ) -> cql.patch[cql.codes.BareQubit]:
        block, _ = cql.ops.measure_pauli(block.data, paulis="Z", record="r")
        block, _ = cql.ops.measure_pauli(block.data,
                                         paulis="Z",
                                         record="ambiguous_alias.r")
        return block

    with pytest.raises(RuntimeError,
                       match="ambiguous measurement record alias"):
        cql.compiler.project_clifford(cql.compile(ambiguous_alias))

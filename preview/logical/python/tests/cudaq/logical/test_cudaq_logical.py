# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import json
from hashlib import sha256
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_device_timing_units_and_operating_point():
    import cudaq.logical as cql

    assert cql.devices.us.nanoseconds == 1_000.0
    timing = cql.devices.TimingModel({"cycle_ns": 2 * cql.devices.us})
    assert timing["cycle_ns"] == 2_000.0
    point = cql.devices.PhysicalOperatingPoint(timing=timing)
    assert point.timing == timing


def test_synthesis_evidence_names_the_shared_cudaq_implementation():
    import cudaq.logical as cql

    @cql.program
    def exact_rotation() -> bool:
        q = cql.rz(cql.prepare_zero(), cql.algebra.pi / 4)
        return cql.measure_z(q)

    build = cql.compile(exact_rotation,
                        pipeline=cql.compiler.pipelines.clifford_t())
    record = next(item for item in build.evidence
                  if item.kind == "logical_gate_set_legalization")
    assert record.producer == "cudaq-synth/gridsynth"


def test_rotated_surface_catalog_entry_is_a_distance_indexed_code():
    import cudaq.logical as cql

    surface_3 = cql.codes.rotated_surface(3)
    assert surface_3 is cql.codes.Surface[3]
    assert (surface_3.n, surface_3.k, surface_3.d.value) == (9, 1, 3)
    assert surface_3.block.size == 17
    assert surface_3.metadata["family"] == "rotated_surface"


def test_product_stage_estimation_and_target_surface():
    import cudaq.logical as cql
    from cudaq.mlir.dialects import _fabric_ops_gen, _qlx_ops_gen
    from cudaq.mlir.dialects import fabric as fabric_dialect
    from cudaq.mlir.dialects import qlx as qlx_dialect

    assert tuple(cql.stages.Stage) == (
        cql.stages.Stage.P0,
        cql.stages.Stage.P1,
        cql.stages.Stage.P2,
        cql.stages.Stage.P3,
    )
    assert tuple(cql.estimate.Tier) == (
        cql.estimate.Tier.LOGICAL,
        cql.estimate.Tier.STATIC,
        cql.estimate.Tier.ANALYTICAL,
        cql.estimate.Tier.SCHEDULE,
    )
    assert hasattr(fabric_dialect, "ResourceRequestOp")
    assert {name for name in vars(_qlx_ops_gen) if name.endswith("Op")
           } == {name for name in vars(qlx_dialect) if name.endswith("Op")}
    assert {name for name in vars(_fabric_ops_gen) if name.endswith("Op")
           } == {name for name in vars(fabric_dialect) if name.endswith("Op")}
    assert "resource" in cql.types.__all__
    authoring_verbs = {
        "allocate_patch",
        "cond",
        "event_await",
        "prepare_plus",
        "produce",
        "request",
        "request_many",
        "resource_rotate",
        "unpack_resource",
        "pack_resource",
        "postselect",
        "xor",
    }
    assert authoring_verbs <= set(cql.ops.__all__)
    for name in authoring_verbs:
        operation = getattr(cql.ops, name)
        assert operation is not None
        assert getattr(cql, name) is operation
    assert isinstance(cql.__version__, str) and cql.__version__
    assert cql.targets.Target.replay(
        cql.targets.mlir.serialize()) is cql.targets.mlir


def test_cudaq_kernel_can_define_a_gadget_objective_and_declaration():
    import cudaq
    import cudaq.logical as cql
    from cudaq.kernel.kernel_decorator import isa_kernel_decorator

    @cudaq.kernel
    def paired_h(left: cudaq.qubit, right: cudaq.qubit):
        h(left)
        h(right)

    @cql.gadget(implements=paired_h)
    def paired_h_gadget(
        left: cql.patch[cql.codes.BareQubit],
        right: cql.patch[cql.codes.BareQubit],
    ) -> tuple[cql.patch[cql.codes.BareQubit], cql.patch[cql.codes.BareQubit]]:
        return cql.ops.h(left.data), cql.ops.h(right.data)

    objective = cql.compile(paired_h_gadget.implements)
    assert objective.root.kind == "action"
    assert objective.to_mlir().count("#qlx.action<h>") == 2

    declaration = paired_h_gadget.kernel
    assert isa_kernel_decorator(declaration)
    declared_function = next(iter(declaration.qkeModule.body.operations))
    assert len(declared_function.operation.regions[0].blocks) == 0
    assert "qlx-objective" in declared_function.operation.attributes

    @cudaq.kernel
    def paired_h_caller():
        left = cudaq.qubit()
        right = cudaq.qubit()
        declaration(left, right)

    imported = cql.compiler.import_cudaq(paired_h_caller)
    assert "qlx.action @paired_h" in imported.to_mlir()
    assert "qlx.apply @paired_h" in imported.to_mlir()


def test_native_logical_objective_exports_a_kernel_through_its_gadget():
    import cudaq
    import cudaq.logical as cql

    @cql.objective
    def paired_x(
        left: cql.types.logical_qubit,
        right: cql.types.logical_qubit,
    ) -> tuple[cql.types.logical_qubit, cql.types.logical_qubit]:
        return cql.ops.x(left), cql.ops.x(right)

    @cql.gadget(implements=paired_x)
    def paired_x_gadget(
        left: cql.patch[cql.codes.BareQubit],
        right: cql.patch[cql.codes.BareQubit],
    ) -> tuple[cql.patch[cql.codes.BareQubit], cql.patch[cql.codes.BareQubit]]:
        return cql.ops.x(left.data), cql.ops.x(right.data)

    assert paired_x_gadget.kernel is paired_x.kernel_declaration
    paired_x_call = paired_x_gadget.kernel

    @cudaq.kernel
    def paired_x_caller():
        left = cudaq.qubit()
        right = cudaq.qubit()
        paired_x_call(left, right)

    imported = cql.compiler.import_cudaq(paired_x_caller)
    assert "qlx.apply @paired_x" in imported.to_mlir()


def test_logical_estimate_is_invariant_under_cached_view_mutation():
    import cudaq.logical as cql
    import cudaq.mlir.ir as mlir_ir

    @cql.program
    def supported_workload() -> bool:
        q = cql.h(cql.prepare_zero())
        return cql.measure_z(q)

    build = cql.compile(supported_workload)
    expected = cql.analysis.logical_counts(build)

    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    cached_action = next(operation for operation in walk(build.module.operation)
                         if operation.name == "qlx.apply")
    cached_action.attributes["action"] = mlir_ir.Attribute.parse(
        "#qlx.action<x>", context=build.module.context)

    assert cql.analysis.logical_counts(build) == expected
    assert "qlx.apply" in build.to_mlir()


def test_p0_resource_requests_are_explicit_and_consumed_linearly():
    import cudaq.logical as cql

    @cql.program
    def p0_workload() -> None:
        qubit = cql.prepare_zero()
        event = cql.ops.request(cql.standard.T_STATE)
        state = cql.ops.event_await(event)
        qubit = cql.ops.consume(
            state,
            qubit,
            action=cql.standard.t,
        )
        cql.discard(qubit)

    build = cql.compile(p0_workload)
    text = build.to_mlir()
    assert 'qlx.resource_request "t_state"' in text
    assert "qlx.consume_resource" in text


def test_retained_workload_and_factory_models_reject_invalid_inputs():
    import cudaq.logical as cql
    import cudaq.logical.algorithms as algorithms

    with pytest.raises(ValueError, match="delta_off == 4"):
        algorithms.gidney_ekera_factor(2048, delta_off=-100)
    with pytest.raises(ValueError, match="input error"):
        cql.protocols.DISTILL_15TO1_T.output_error(-0.1)
    with pytest.raises(ValueError, match="input error"):
        cql.protocols.DISTILL_15TO1_T.acceptance_probability(1.1)
    model = cql.protocols.DISTILL_15TO1_T
    assert model.acceptance_probability(
        1.0 / 15.0) == pytest.approx(0.3608925819844537)
    assert model.output_error(1.0 / 15.0) == pytest.approx(0.012867756815146294)
    assert model.output_error(1.0e-9) == pytest.approx(3.5e-26, rel=4.0e-9)
    near_one = 1.0 - 2.0**-24
    assert 0.0 <= model.output_error(near_one) <= 1.0
    assert model.output_error(near_one) == pytest.approx(
        1.0 - model.output_error(1.0 - near_one))


def test_ordinary_gadget_result_role_round_trips_through_native_verification():
    import cudaq.logical as cql
    from cudaq.logical.gadgets import OutcomeRole

    measurement = cql.gadgets.logical_measure(cql.codes.rotated_surface(3),
                                              basis="z")
    build = cql.compile(measurement)
    assert measurement.outcome_map.indices_for(OutcomeRole.RESULT) == (0,)
    compact_mlir = "".join(build.to_mlir().split())
    assert 'roles=[["result"]]' in compact_mlir
    assert build.module.operation.verify()


def test_distillation_peak_counts_unpacked_resource_payloads():
    import cudaq.logical as cql

    build = cql.compile(cql.protocols.distill_15to1)
    counts = cql.estimate(build, tier=cql.estimate.Tier.STATIC)
    assert cql.protocols.distill_15to1.metadata[
        "production_model"] == "distill-15to1-T"
    assert counts.patches_peak == 5
    assert counts.logical_qubits_peak == 5
    assert tuple(facet.value for facet in build.facets) == (
        "qec_spec",
        "protocol_network",
    )
    assert counts.source_facets == tuple(facet.value for facet in build.facets)


def test_physical_mpp_rejects_duplicate_carriers_before_emission():
    import cudaq.logical as cql

    @cql.gadget(implements=cql.std.idle)
    def duplicate(
        block: cql.patch[cql.codes.BareQubit],
    ) -> cql.patch[cql.codes.BareQubit]:
        block, _ = cql.ops.measure_pauli(block.data[(0, 0)], paulis="XZ")
        return block

    with pytest.raises(ValueError, match="distinct physical carrier"):
        cql.compile(duplicate)


def test_result_and_build_provenance_fail_closed():
    import cudaq.logical as cql

    p1 = runpy.run_path(
        str(ROOT / "examples/standalone/01_logical_placement.py"))["placed"]
    serialized = bytearray(p1.serialize())
    serialized[-1] ^= 1
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        cql.compiler.Build.replay(bytes(serialized))

    p1_bundle = json.loads(p1.serialize())
    old_placement_digest = "sha256:" + sha256(
        json.dumps(p1_bundle["placement"],
                   sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    p1_bundle["placement"]["bindings"][1]["slot"] = 0
    p1_bundle["experiment"]["bindings"]["placement"] = p1_bundle["placement"]
    new_placement_digest = "sha256:" + sha256(
        json.dumps(p1_bundle["placement"],
                   sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    p1_bundle["module"] = p1_bundle["module"].replace(old_placement_digest,
                                                      new_placement_digest)
    from cudaq.logical.compiler.build_bundle import _build_bundle_content_sha256
    p1_bundle["content_sha256"] = _build_bundle_content_sha256(p1_bundle)
    with pytest.raises(ValueError, match="placement ownership facts differ"):
        cql.compiler.Build.replay(
            json.dumps(p1_bundle, sort_keys=True,
                       separators=(",", ":")).encode())

    @cql.program
    def p2_fixture() -> bool:
        return cql.measure_z(cql.prepare_zero())

    surface_3 = cql.codes.Surface[3]
    device_builder = cql.devices.DeviceBuilder("ReplaySurfaceDevice")
    compute = device_builder.logical.add_compute(capacity=1)
    device_builder.qec.bind(compute, encoding=surface_3)
    device = device_builder.build()
    p0 = cql.compile(p2_fixture)
    p1 = cql.compiler.place(p0, device=device)
    p2 = cql.compile(
        p1,
        pipeline=cql.compiler.pipelines.qec(),
        device=device,
    )
    bundle = json.loads(p2.serialize())
    old_selection_digest = "sha256:" + sha256(
        json.dumps(bundle["qec_selection"],
                   sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    bundle["qec_selection"]["actions"][0]["selected"] = "forged"
    bundle["qec_selection"]["actions"][0]["feasible_candidates"] = ["forged"]
    new_selection_digest = "sha256:" + sha256(
        json.dumps(bundle["qec_selection"],
                   sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    bundle["module"] = bundle["module"].replace(old_selection_digest,
                                                new_selection_digest)
    bundle["content_sha256"] = _build_bundle_content_sha256(bundle)
    with pytest.raises(ValueError,
                       match="selected realization is not retained"):
        cql.compiler.Build.replay(
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode())

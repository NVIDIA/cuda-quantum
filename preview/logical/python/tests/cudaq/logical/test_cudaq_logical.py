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
    import cudaq.logical as ql

    assert ql.devices.us.nanoseconds == 1_000.0
    timing = ql.devices.TimingModel({"cycle_ns": 2 * ql.devices.us})
    assert timing["cycle_ns"] == 2_000.0
    point = ql.devices.PhysicalOperatingPoint(timing=timing)
    assert point.timing == timing


def test_synthesis_evidence_names_the_shared_cudaq_implementation():
    import cudaq.logical

    @cudaq.logical.program
    def exact_rotation() -> bool:
        q = cudaq.logical.rz(cudaq.logical.prepare_zero(),
                             cudaq.logical.algebra.pi / 4)
        return cudaq.logical.measure_z(q)

    build = cudaq.logical.compile(
        exact_rotation, pipeline=cudaq.logical.compiler.pipelines.clifford_t())
    record = next(item for item in build.evidence
                  if item.kind == "logical_gate_set_legalization")
    assert record.producer == "cudaq-synth/gridsynth"


def test_rotated_surface_catalog_entry_is_a_distance_indexed_code():
    import cudaq.logical

    surface_3 = cudaq.logical.codes.rotated_surface(3)
    assert surface_3 is cudaq.logical.codes.Surface[3]
    assert (surface_3.n, surface_3.k, surface_3.d.value) == (9, 1, 3)
    assert surface_3.block.size == 17
    assert surface_3.metadata["family"] == "rotated_surface"


def test_product_stage_estimation_and_target_surface():
    import cudaq.logical
    from cudaq.mlir.dialects import _fabric_ops_gen, _qlx_ops_gen
    from cudaq.mlir.dialects import fabric as fabric_dialect
    from cudaq.mlir.dialects import qlx as qlx_dialect

    assert tuple(cudaq.logical.stages.Stage) == (
        cudaq.logical.stages.Stage.P0,
        cudaq.logical.stages.Stage.P1,
        cudaq.logical.stages.Stage.P2,
        cudaq.logical.stages.Stage.P3,
    )
    assert tuple(cudaq.logical.estimate.Tier) == (
        cudaq.logical.estimate.Tier.LOGICAL,
        cudaq.logical.estimate.Tier.STATIC,
        cudaq.logical.estimate.Tier.ANALYTICAL,
        cudaq.logical.estimate.Tier.SCHEDULE,
    )
    assert hasattr(fabric_dialect, "ResourceRequestOp")
    assert {name for name in vars(_qlx_ops_gen) if name.endswith("Op")
           } == {name for name in vars(qlx_dialect) if name.endswith("Op")}
    assert {name for name in vars(_fabric_ops_gen) if name.endswith("Op")
           } == {name for name in vars(fabric_dialect) if name.endswith("Op")}
    assert "resource" in cudaq.logical.types.__all__
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
    assert authoring_verbs <= set(cudaq.logical.ops.__all__)
    for name in authoring_verbs:
        operation = getattr(cudaq.logical.ops, name)
        assert operation is not None
        assert getattr(cudaq.logical, name) is operation
    assert isinstance(cudaq.logical.__version__,
                      str) and cudaq.logical.__version__
    assert cudaq.logical.targets.Target.replay(
        cudaq.logical.targets.mlir.serialize()) is cudaq.logical.targets.mlir


def test_cudaq_kernel_can_define_a_gadget_objective_and_declaration():
    import cudaq
    import cudaq.logical as ql
    from cudaq.kernel.kernel_decorator import isa_kernel_decorator

    @cudaq.kernel
    def paired_h(left: cudaq.qubit, right: cudaq.qubit):
        h(left)
        h(right)

    @ql.gadget(implements=paired_h)
    def paired_h_gadget(
        left: ql.patch[ql.codes.BareQubit],
        right: ql.patch[ql.codes.BareQubit],
    ) -> tuple[ql.patch[ql.codes.BareQubit], ql.patch[ql.codes.BareQubit]]:
        return ql.ops.h(left.data), ql.ops.h(right.data)

    objective = ql.compile(paired_h_gadget.implements)
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

    imported = ql.compiler.import_cudaq(paired_h_caller)
    assert "qlx.action @paired_h" in imported.to_mlir()
    assert "qlx.apply @paired_h" in imported.to_mlir()


def test_native_logical_objective_exports_a_kernel_through_its_gadget():
    import cudaq
    import cudaq.logical as ql

    @ql.objective
    def paired_x(
        left: ql.types.logical_qubit,
        right: ql.types.logical_qubit,
    ) -> tuple[ql.types.logical_qubit, ql.types.logical_qubit]:
        return ql.ops.x(left), ql.ops.x(right)

    @ql.gadget(implements=paired_x)
    def paired_x_gadget(
        left: ql.patch[ql.codes.BareQubit],
        right: ql.patch[ql.codes.BareQubit],
    ) -> tuple[ql.patch[ql.codes.BareQubit], ql.patch[ql.codes.BareQubit]]:
        return ql.ops.x(left.data), ql.ops.x(right.data)

    assert paired_x_gadget.kernel is paired_x.kernel_declaration
    paired_x_call = paired_x_gadget.kernel

    @cudaq.kernel
    def paired_x_caller():
        left = cudaq.qubit()
        right = cudaq.qubit()
        paired_x_call(left, right)

    imported = ql.compiler.import_cudaq(paired_x_caller)
    assert "qlx.apply @paired_x" in imported.to_mlir()


def test_logical_estimate_is_invariant_under_cached_view_mutation():
    import cudaq.logical

    @cudaq.logical.program
    def supported_workload() -> bool:
        q = cudaq.logical.h(cudaq.logical.prepare_zero())
        return cudaq.logical.measure_z(q)

    build = cudaq.logical.compile(supported_workload)
    expected = cudaq.logical.analysis.logical_counts(build)

    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    cached_action = next(operation for operation in walk(build.module.operation)
                         if operation.name == "qlx.apply")
    cached_action.attributes["action"] = cudaq.mlir.ir.Attribute.parse(
        "#qlx.action<x>", context=build.module.context)

    assert cudaq.logical.analysis.logical_counts(build) == expected
    assert "qlx.apply" in build.to_mlir()


def test_p0_resource_requests_are_explicit_and_consumed_linearly():
    import cudaq.logical

    @cudaq.logical.program
    def p0_workload() -> None:
        qubit = cudaq.logical.prepare_zero()
        event = cudaq.logical.ops.request(cudaq.logical.standard.T_STATE)
        state = cudaq.logical.ops.event_await(event)
        qubit = cudaq.logical.ops.consume(
            state,
            qubit,
            action=cudaq.logical.standard.t,
        )
        cudaq.logical.discard(qubit)

    build = cudaq.logical.compile(p0_workload)
    text = build.to_mlir()
    assert 'qlx.resource_request "t_state"' in text
    assert "qlx.consume_resource" in text


def test_retained_workload_and_factory_models_reject_invalid_inputs():
    import cudaq.logical
    import cudaq.logical.algorithms as algorithms

    with pytest.raises(ValueError, match="delta_off == 4"):
        algorithms.gidney_ekera_factor(2048, delta_off=-100)
    with pytest.raises(ValueError, match="input error"):
        cudaq.logical.protocols.DISTILL_15TO1_T.output_error(-0.1)
    with pytest.raises(ValueError, match="input error"):
        cudaq.logical.protocols.DISTILL_15TO1_T.acceptance_probability(1.1)
    model = cudaq.logical.protocols.DISTILL_15TO1_T
    assert model.acceptance_probability(
        1.0 / 15.0) == pytest.approx(0.3608925819844537)
    assert model.output_error(1.0 / 15.0) == pytest.approx(0.012867756815146294)
    assert model.output_error(1.0e-9) == pytest.approx(3.5e-26, rel=4.0e-9)
    near_one = 1.0 - 2.0**-24
    assert 0.0 <= model.output_error(near_one) <= 1.0
    assert model.output_error(near_one) == pytest.approx(
        1.0 - model.output_error(1.0 - near_one))


def test_ordinary_gadget_result_role_round_trips_through_native_verification():
    import cudaq.logical
    from cudaq.logical.gadgets import OutcomeRole

    measurement = cudaq.logical.gadgets.logical_measure(
        cudaq.logical.codes.rotated_surface(3), basis="z")
    build = cudaq.logical.compile(measurement)
    assert measurement.outcome_map.indices_for(OutcomeRole.RESULT) == (0,)
    compact_mlir = "".join(build.to_mlir().split())
    assert 'roles=[["result"]]' in compact_mlir
    assert build.module.operation.verify()


def test_distillation_peak_counts_unpacked_resource_payloads():
    import cudaq.logical

    build = cudaq.logical.compile(cudaq.logical.protocols.distill_15to1)
    counts = cudaq.logical.estimate(build,
                                    tier=cudaq.logical.estimate.Tier.STATIC)
    assert cudaq.logical.protocols.distill_15to1.metadata[
        "production_model"] == "distill-15to1-T"
    assert counts.patches_peak == 5
    assert counts.logical_qubits_peak == 5
    assert tuple(facet.value for facet in build.facets) == (
        "qec_spec",
        "protocol_network",
    )
    assert counts.source_facets == tuple(facet.value for facet in build.facets)


def test_physical_mpp_rejects_duplicate_carriers_before_emission():
    import cudaq.logical

    @cudaq.logical.gadget(implements=cudaq.logical.std.idle)
    def duplicate(
        block: cudaq.logical.patch[cudaq.logical.codes.BareQubit],
    ) -> cudaq.logical.patch[cudaq.logical.codes.BareQubit]:
        block, _ = cudaq.logical.ops.measure_pauli(block.data[(0, 0)],
                                                   paulis="XZ")
        return block

    with pytest.raises(ValueError, match="distinct physical carrier"):
        cudaq.logical.compile(duplicate)


def test_result_and_build_provenance_fail_closed():
    import cudaq.logical

    p1 = runpy.run_path(
        str(ROOT / "examples/standalone/01_logical_placement.py"))["placed"]
    serialized = bytearray(p1.serialize())
    serialized[-1] ^= 1
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        cudaq.logical.compiler.Build.replay(bytes(serialized))

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
        cudaq.logical.compiler.Build.replay(
            json.dumps(p1_bundle, sort_keys=True,
                       separators=(",", ":")).encode())

    @cudaq.logical.program
    def p2_fixture() -> bool:
        return cudaq.logical.measure_z(cudaq.logical.prepare_zero())

    surface_3 = cudaq.logical.codes.Surface[3]
    device_builder = cudaq.logical.devices.DeviceBuilder("ReplaySurfaceDevice")
    compute = device_builder.logical.add_compute(capacity=1)
    device_builder.qec.bind(compute, encoding=surface_3)
    device = device_builder.build()
    p0 = cudaq.logical.compile(p2_fixture)
    p1 = cudaq.logical.compiler.place(p0, device=device)
    p2 = cudaq.logical.compile(
        p1,
        pipeline=cudaq.logical.compiler.pipelines.qec(),
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
        cudaq.logical.compiler.Build.replay(
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode())

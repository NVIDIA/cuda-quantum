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
EXAMPLES = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))


def test_examples_execute(capsys):
    assert len(EXAMPLES) == 8
    output = {}
    for example in EXAMPLES:
        runpy.run_path(str(example), run_name="__main__")
        output[example.name] = capsys.readouterr().out
    assert "Gidney--Ekerå @cudaq.kernel -> cudaq.estimate:" in output[
        "06_gidney_ekera.py"]


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
    from cudaq.logical.dialects import _fabric_ops_gen, _qlx_ops_gen
    from cudaq.logical.dialects import fabric as fabric_dialect
    from cudaq.logical.dialects import qlx as qlx_dialect

    assert tuple(cudaq.logical.stages.Stage) == (
        cudaq.logical.stages.Stage.P0,
        cudaq.logical.stages.Stage.P1,
        cudaq.logical.stages.Stage.P2,
    )
    assert tuple(cudaq.logical.estimate.Tier) == (
        cudaq.logical.estimate.Tier.LOGICAL,
        cudaq.logical.estimate.Tier.STATIC,
    )
    assert hasattr(fabric_dialect, "ResourceRequestOp")
    assert {name for name in vars(_qlx_ops_gen) if name.endswith("Op")
           } == {name for name in qlx_dialect.__all__ if name.endswith("Op")}
    assert {name for name in vars(_fabric_ops_gen) if name.endswith("Op")} == {
        name for name in fabric_dialect.__all__ if name.endswith("Op")
    }
    assert "resource" in cudaq.logical.types.__all__
    authoring_verbs = {
        "allocate_patch",
        "prepare_plus",
        "request_many",
        "unpack_resource",
        "pack_resource",
        "postselect",
    }
    assert authoring_verbs <= set(cudaq.logical.__all__)
    for name in authoring_verbs:
        assert getattr(cudaq.logical, name) is getattr(cudaq.logical.ops, name)
    assert isinstance(cudaq.logical.__version__,
                      str) and cudaq.logical.__version__
    assert cudaq.logical.targets.Target.replay(
        cudaq.logical.targets.mlir.serialize()) is cudaq.logical.targets.mlir


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
    cached_action.attributes["action"] = cudaq.logical.ir.Attribute.parse(
        "#qlx.action<x>", context=build.module.context)

    assert cudaq.logical.analysis.logical_counts(build) == expected
    assert "qlx.apply" in build.to_mlir()


def test_resource_requests_are_owned_by_p2_protocol_authoring():
    import cudaq.logical

    @cudaq.logical.program
    def p0_workload() -> None:
        cudaq.logical.ops.request(cudaq.logical.std.T_STATE)

    with pytest.raises(TypeError,
                       match="only inside an @cudaq.logical.protocol body"):
        cudaq.logical.compile(p0_workload)


def test_retained_workload_and_factory_models_reject_invalid_inputs():
    import cudaq.logical
    from cudaq.logical import algorithms

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

    measurement = cudaq.logical.gadgets.logical_measure(
        cudaq.logical.codes.rotated_surface(3), basis="z")
    build = cudaq.logical.compile(measurement)
    compact_mlir = "".join(build.to_mlir().split())
    assert 'roles=[["result"]]' in compact_mlir
    assert build.module.operation.verify()


def test_distillation_peak_counts_unpacked_resource_payloads():
    import cudaq.logical

    build = cudaq.logical.compile(cudaq.logical.protocols.distill_15to1)
    counts = cudaq.logical.estimate(build,
                                    tier=cudaq.logical.estimate.Tier.STATIC)
    assert not cudaq.logical.protocols.distill_15to1.metadata
    assert counts.patches_peak == 5
    assert counts.logical_qubits_peak == 5
    assert tuple(facet.value for facet in build.facets) == (
        "qec_spec",
        "qec_realization",
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

    p1 = runpy.run_path(str(ROOT / "examples/02_p1_placement.py"))["p1"]
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

    endpoint = cudaq.logical.targets.surface_target(
        logical_capacity=1).runtime_endpoint
    p0 = endpoint.compile(cudaq.logical.compile(p2_fixture))
    p0 = endpoint.next_backend.compile(p0)
    p1 = endpoint.next_backend.next_backend.compile(p0)
    p2 = endpoint.next_backend.next_backend.next_backend.compile(p1)
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

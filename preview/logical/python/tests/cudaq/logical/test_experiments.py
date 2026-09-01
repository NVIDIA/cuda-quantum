# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import json

import pytest

import cudaq.logical


def test_definitions_attach_to_one_user_owned_mlir_module():
    owner = cudaq.logical.compiler.CompilationContext()
    module = owner.module

    @cudaq.logical.program
    def first() -> bool:
        return cudaq.logical.measure_z(cudaq.logical.prepare_zero())

    @cudaq.logical.program
    def second() -> bool:
        return cudaq.logical.measure_x(cudaq.logical.ops.prepare_plus())

    first_build = cudaq.logical.compile(first, module=module)
    cudaq.logical.compile(cudaq.logical.codes.Steane, module=module)
    second_build = cudaq.logical.compile(second, module=module)
    cudaq.logical.targets.mlir.materialize(module=module)

    text = str(module)
    assert text.count("qlx.program @first") == 1
    assert text.count("qlx.program @second") == 1
    assert "fabric.code @Steane" in text
    assert "qlx.target_manifest @mlir" in text
    assert "qlx.experiment" not in text
    assert second_build.experiment.root == second_build.root
    assert "qlx.program @second" not in first_build.to_mlir()
    assert "qlx.program @first" in second_build.to_mlir()


@cudaq.logical.program
def experiment_memory() -> bool:
    q = cudaq.logical.prepare_zero()
    q = cudaq.logical.idle(q, rounds=3)
    return cudaq.logical.measure_z(q)


def test_direct_and_explicit_experiment_requests_normalize_identically():
    direct = cudaq.logical.compile(experiment_memory, parameters={"bias": 10.0})
    explicit = cudaq.logical.compile(
        cudaq.logical.compiler.Experiment(root=experiment_memory,
                                          parameters={"bias": 10.0}))

    assert direct.to_mlir() == explicit.to_mlir()
    assert direct.experiment.to_bundle() == explicit.experiment.to_bundle()
    assert direct.experiment.root == direct.root
    assert direct.experiment.stage == cudaq.logical.stages.P0
    assert direct.experiment.facets == ()
    assert direct.experiment.bindings() == {"parameters": {"bias": 10.0}}
    assert "qlx.experiment" not in direct.to_mlir()


def test_experiment_survives_a_clean_build_replay():
    build = cudaq.logical.compile(
        cudaq.logical.compiler.Experiment(
            root=experiment_memory,
            policy={"evidence": "require"},
            parameters={"rounds": 3},
        ))
    replayed = cudaq.logical.compiler.Build.replay(build.serialize())

    assert replayed.experiment.to_bundle() == build.experiment.to_bundle()
    envelope = json.loads(build.serialize())
    assert envelope["experiment"] == build.experiment.to_bundle()
    assert replayed.verify()


def test_placement_callback_is_eliminated_to_an_exact_witness():

    @cudaq.logical.machine
    class OneSlot:
        memory = cudaq.logical.architecture.Space(capacity=1)

    placed = cudaq.logical.compile(
        experiment_memory,
        pipeline=cudaq.logical.compiler.pipelines.placed(),
        device=OneSlot,
        placement=lambda values: (),
    )

    bindings = placed.experiment.bindings()
    assert bindings["device"]["name"] == "OneSlot"
    assert bindings["placement"]["machine"] == "OneSlot"
    assert bindings["placement"]["bindings"][0]["space"] == "memory"
    assert "function" not in json.dumps(bindings)
    assert cudaq.logical.compiler.Build.replay(
        placed.serialize()).experiment.to_bundle() == (
            placed.experiment.to_bundle())


def test_compile_many_is_an_immutable_self_describing_sweep_bundle():
    points = tuple(
        cudaq.logical.compiler.Experiment(root=experiment_memory,
                                          parameters={"p": p})
        for p in (1e-4, 1e-3, 1e-2))
    bundle = cudaq.logical.compiler.compile_many(
        points, pipeline=cudaq.logical.compiler.pipelines.logical())

    assert isinstance(bundle, cudaq.logical.compiler.ExperimentBundle)
    assert len(bundle) == 3
    assert [build.experiment.bindings()["parameters"]["p"] for build in bundle
           ] == [
               1e-4,
               1e-3,
               1e-2,
           ]
    replayed = cudaq.logical.compiler.ExperimentBundle.replay(
        bundle.serialize())
    assert [build.to_mlir() for build in replayed
           ] == [build.to_mlir() for build in bundle]
    assert [item.to_bundle() for item in replayed.experiments
           ] == [item.to_bundle() for item in bundle.experiments]


def test_unsupported_profile_routes_fail_as_invalid_pipelines_not_placeholders(
):
    p0 = cudaq.logical.compile(experiment_memory)
    with pytest.raises(
            ValueError,
            match="no CUDA-Q Logical compilation route from 'p0' to 'p2s'"):
        cudaq.logical.compile(
            p0, pipeline=cudaq.logical.compiler.pipelines.qec_definitions())

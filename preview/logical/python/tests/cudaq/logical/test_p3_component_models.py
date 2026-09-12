# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fast public-path checks for compact P3 physical component models."""

from dataclasses import replace

import pytest

from cudaq import logical
from cudaq.logical.architectures import surface

SURFACE = surface.definitions(3)
CODE = SURFACE.code


@logical.protocol(implements=logical.logical.produce(logical.standard.T_STATE))
def selected_t_factory() -> logical.types.resource[logical.standard.T_STATE]:
    output = SURFACE.prepare_plus(
        logical.ops.allocate_patch(CODE, region="factory"))
    check = SURFACE.prepare_zero(
        logical.ops.allocate_patch(CODE, region="factory"))
    logical.ops.postselect(SURFACE.measure_z(check), expected=False)
    return logical.ops.pack_resource(output, kind=logical.standard.T_STATE)


def _timing():
    return {
        "surface_cycle_ns": 1_000.0,
        "h_ns": 3_000.0,
        "x_ns": 1_000.0,
        "z_ns": 1_000.0,
        "measure_x_instrument_ns": 2_000.0,
        "measure_z_instrument_ns": 2_000.0,
    }


def _factory_device(*, model=None, patch_count=3):
    builder = logical.devices.DeviceBuilder("SelectedTFactoryDevice")
    factory = builder.logical.add_factory(
        produces=logical.standard.T_STATE,
        via=selected_t_factory,
        capacity=1,
        name="factory",
        stream_name="t_states",
    )
    factory_qec = builder.qec.bind(
        factory,
        architecture=logical.devices.QECArchitecture("selected_t_factory_d3",
                                                     CODE.default_encoding),
        block_capacity=2,
    )
    resources = builder.physical.add_resources(
        "surface_code_patch",
        patch_count,
        name="factory_patches",
        granularity=logical.architecture.ResourceGranularity.PATCH,
        footprint=SURFACE.square_patch_footprint,
        native_actions=logical.architecture.physical_actions.clifford_set(),
        native_instruments=(
            logical.architecture.physical_instruments.MX,
            logical.architecture.physical_instruments.MZ,
        ),
    )
    builder.physical.bind(factory_qec, to=resources, factory_model=model)
    builder.physical.set_operating_point(timing=_timing())
    return builder.build()


def test_p3_compiler_mints_opaque_factory_characterization():
    device = _factory_device()
    p3 = logical.compile(
        selected_t_factory,
        pipeline=logical.compiler.pipelines.physical(),
        device=device,
    )
    schedule = logical.compiler.schedule(p3)
    model = logical.compiler.factory_model(schedule,
                                           produces=logical.standard.T_STATE)
    characterization = model.characterization

    assert p3.stage == logical.stages.P3
    assert schedule.build.module.operation.verify()
    assert characterization is not None
    assert characterization.resource_kind == logical.standard.T_STATE
    assert characterization.source_provider == "selected_t_factory"
    assert characterization.code_distances == (3,)
    assert len(characterization.output_events) == 1
    assert len(characterization.selection_events) == 1

    rebound = _factory_device(model=model)
    assert rebound.qec_to_physical[0].factory_model is model

    with pytest.raises(TypeError, match="opaque compiler artifact"):
        replace(characterization, physical_units=1)
    with pytest.raises(ValueError, match="exact compiler-derived startup"):
        replace(model, startup_cycles=0.5, output_interval_cycles=0.5)


def test_p3_component_model_constructors_fail_closed():
    links = logical.architecture.ResourceClass("transport_lane",
                                               4,
                                               name="links")
    claim = logical.devices.PhysicalResourceClaim(links)
    evidence = logical.analysis.user_assertion("constructor test")

    with pytest.raises(TypeError, match="finite positive"):
        logical.devices.TransportModel(
            latency_cycles=0.0,
            initiation_interval_cycles=1.0,
            resources=(claim,),
            endpoint_occupancy=(1, 1),
            evidence=evidence,
        )

    with pytest.raises(ValueError, match="backpressured interval semantics"):
        logical.devices.TransportModel(
            latency_cycles=1.0,
            initiation_interval_cycles=2.0,
            resources=(claim,),
            endpoint_occupancy=(1, 1),
            evidence=evidence,
        )

    phase = logical.devices.SpacetimePhase(
        "only",
        steps=1,
        step_duration_cycles=1.0,
        resources=(claim,),
    )
    with pytest.raises(ValueError, match="guaranteed or single_shot"):
        logical.devices.SpacetimePlanModel(
            protocol=selected_t_factory,
            latency_cycles=1.0,
            initiation_interval_cycles=1.0,
            phases=(phase,),
            evidence=evidence,
            policy="expected_retry_latency",
        )

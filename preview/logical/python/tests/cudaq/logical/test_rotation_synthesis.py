# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Focused generated-rotation specialization coverage."""

import cudaq.logical as cql
from cudaq.logical.qec.steane import pygridsynth_rpp  # noqa: F401

steane_synthesis_builder = cql.devices.DeviceBuilder("SteaneSynthesisDevice")
steane_compute = steane_synthesis_builder.logical.add_compute(capacity=2)
steane_synthesis_builder.logical.add_stream(
    cql.standard.RAW_T_STATE,
    name="raw_magic",
    buffer_size=15,
    external=True,
)
steane_synthesis_builder.qec.bind(
    steane_compute,
    encoding=cql.codes.Steane,
)
STEANE_SYNTHESIS_DEVICE = steane_synthesis_builder.build()


@cql.program
def negative_exact_t_rotation() -> None:
    data = cql.allocate(1, state=cql.types.zero)
    data[0], = cql.ops.rotate(
        cql.types.Z(data[0]),
        angle=-cql.algebra.pi / 4,
        precision=1e-12,
    )
    cql.discard(data)


def test_negative_symbolic_exact_rotation_records_signed_specialization():
    build = cql.compile(
        negative_exact_t_rotation,
        pipeline=cql.compiler.pipelines.qec(),
        device=STEANE_SYNTHESIS_DEVICE,
    )
    text = build.to_mlir()

    assert 'rpp_strategy = "t_injection"' in text
    assert "angle_pi_numer = -1 : i64" in text
    assert "angle_pi_denom = 4 : i64" in text
    assert "fabric.call @steane_t_from_factory" in text
    assert build.module.operation.verify()

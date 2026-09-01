# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import pytest

import cudaq.logical


def test_standard_y_state_consumes_with_logical_s():
    assert cudaq.logical.std.Y_STATE == cudaq.logical.types.ResourceKind(
        "y_state",
        consume_action=cudaq.logical.std.s,
    )
    assert cudaq.logical.std.Y_STATE.consume_action is cudaq.logical.std.s


def test_produce_uses_typed_region_and_enclosing_protocol_symbol():
    builder = cudaq.logical.devices.DeviceBuilder("TypedFactoryDevice")
    factory = builder.logical.add_region("factory", capacity=1)

    @cudaq.logical.protocol(
        implements=cudaq.logical.std.produce(cudaq.logical.std.Y_STATE),
        name="typed_y_factory",
    )
    def typed_y_factory(
    ) -> cudaq.logical.types.resource[cudaq.logical.std.Y_STATE]:
        return cudaq.logical.ops.produce(cudaq.logical.std.Y_STATE,
                                         region=factory)

    build = cudaq.logical.compile(typed_y_factory)
    text = build.to_mlir()

    assert "fabric.produce_resource" in text
    assert "region = @factory" in text
    assert "protocol = @typed_y_factory" in text
    assert "#fabric.spec_only" not in text
    assert build.module.operation.verify()


def test_produce_rejects_raw_string_region():

    @cudaq.logical.protocol(
        implements=cudaq.logical.std.produce(cudaq.logical.std.Y_STATE),
        name="raw_string_y_factory",
    )
    def raw_string_y_factory(
    ) -> cudaq.logical.types.resource[cudaq.logical.std.Y_STATE]:
        return cudaq.logical.ops.produce(cudaq.logical.std.Y_STATE,
                                         region="factory")

    with pytest.raises(
            TypeError,
            match="region= requires a typed logical region",
    ):
        cudaq.logical.compile(raw_string_y_factory)


def test_concrete_y_factory_authors_and_packs_its_encoded_payload():
    builder = cudaq.logical.devices.DeviceBuilder("ConcreteFactoryDevice")
    factory = builder.logical.add_region("factory", capacity=1)
    encoding = cudaq.logical.codes.Surface[3]
    prepare_encoded_y = cudaq.logical.gadgets.stabilizer_preparation(
        encoding,
        logical_stabilizers=(cudaq.logical.types.Y(0),),
        name="prepare_concrete_encoded_y",
    )

    @cudaq.logical.protocol(implements=cudaq.logical.std.produce(
        cudaq.logical.std.Y_STATE),)
    def concrete_y_factory(
    ) -> cudaq.logical.types.resource[cudaq.logical.std.Y_STATE]:
        payload = cudaq.logical.ops.allocate_patch(encoding, region=factory)
        payload = prepare_encoded_y(payload)
        return cudaq.logical.ops.pack_resource(
            payload,
            kind=cudaq.logical.std.Y_STATE,
        )

    text = cudaq.logical.compile(concrete_y_factory).to_mlir()

    assert "fabric.alloc" in text
    assert "region = @factory" in text
    assert "fabric.call @prepare_concrete_encoded_y" in text
    assert "fabric.pack_resource" in text
    assert "as @y_state" in text
    assert "fabric.produce_resource" not in text

    with pytest.raises(AttributeError, match="immutable"):
        concrete_y_factory.provider = lambda: None
    with pytest.raises(AttributeError, match="immutable"):
        del concrete_y_factory.provider
    with pytest.raises(AttributeError, match="immutable"):
        prepare_encoded_y.provider = lambda block: block


def test_p2_gadgets_compose_through_an_ordinary_nested_call():
    steane = cudaq.logical.codes.Steane
    inner = cudaq.logical.gadgets.logical_pauli(
        steane,
        basis="x",
        name="nested_steane_logical_x",
    )

    @cudaq.logical.gadget(implements=inner.implements)
    def outer(
        block: cudaq.logical.patch[steane],) -> cudaq.logical.patch[steane]:
        return inner(block)

    text = cudaq.logical.compile(outer).to_mlir()

    assert "fabric.call @nested_steane_logical_x" in text
    assert "fabric.x" in text

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import pytest

import cudaq.logical as qlx
from cudaq.logical.std import LogicalInstrumentRef


def test_standard_y_state_consumes_with_logical_s():
    assert qlx.standard.Y_STATE == qlx.types.ResourceKind(
        "y_state",
        consume_action=qlx.logical.s,
    )
    assert qlx.standard.Y_STATE.consume_action is qlx.logical.s


def test_produce_uses_typed_region_and_enclosing_protocol_symbol():
    builder = qlx.devices.DeviceBuilder("TypedFactoryDevice")
    factory = builder.logical.add_region("factory", capacity=1)

    @qlx.protocol(
        implements=qlx.logical.produce(qlx.standard.Y_STATE),
        name="typed_y_factory",
    )
    def typed_y_factory() -> qlx.types.resource[qlx.standard.Y_STATE]:
        return qlx.ops.produce(qlx.standard.Y_STATE, region=factory)

    build = qlx.compile(typed_y_factory)
    text = build.to_mlir()

    assert "fabric.produce_resource" in text
    assert "region = @factory" in text
    assert "protocol = @typed_y_factory" in text
    assert "#fabric.spec_only" not in text
    assert build.module.operation.verify()


def test_produce_rejects_raw_string_region():

    @qlx.protocol(
        implements=qlx.logical.produce(qlx.standard.Y_STATE),
        name="raw_string_y_factory",
    )
    def raw_string_y_factory() -> qlx.types.resource[qlx.standard.Y_STATE]:
        return qlx.ops.produce(qlx.standard.Y_STATE, region="factory")

    with pytest.raises(
            TypeError,
            match="region= requires a typed logical region",
    ):
        qlx.compile(raw_string_y_factory)


def test_concrete_y_factory_authors_and_packs_its_encoded_payload():
    builder = qlx.devices.DeviceBuilder("ConcreteFactoryDevice")
    factory = builder.logical.add_region("factory", capacity=1)
    encoding = qlx.codes.Surface[3]

    @qlx.gadget(
        implements=LogicalInstrumentRef("prepare_y", 0, 1),
        name="prepare_encoded_y",
    )
    def prepare_encoded_y(block: qlx.patch[encoding],) -> qlx.patch[encoding]:
        # S H |0> = |+i>, the positive Y eigenstate. This belongs in a
        # gadget: a protocol composes typed realizations but does not emit
        # physical logical operations directly.
        block = qlx.h(block.data)
        return qlx.s(block.data)

    @qlx.protocol(implements=qlx.logical.produce(qlx.standard.Y_STATE),)
    def concrete_y_factory() -> qlx.types.resource[qlx.standard.Y_STATE]:
        payload = qlx.ops.allocate_patch(encoding, region=factory)
        payload = prepare_encoded_y(payload)
        return qlx.ops.pack_resource(
            payload,
            kind=qlx.standard.Y_STATE,
        )

    text = qlx.compile(concrete_y_factory).to_mlir()

    assert "fabric.alloc" in text
    assert "region = @factory" in text
    assert "fabric.call @prepare_encoded_y" in text
    assert "fabric.h" in text
    assert "fabric.s" in text
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
    steane = qlx.codes.Steane
    inner = qlx.gadgets.logical_pauli(
        steane,
        basis="x",
        name="nested_steane_logical_x",
    )

    @qlx.gadget(implements=inner.implements)
    def outer(block: qlx.patch[steane],) -> qlx.patch[steane]:
        return inner(block)

    text = qlx.compile(outer).to_mlir()
    assert "fabric.call @nested_steane_logical_x" in text
    assert "fabric.x" in text

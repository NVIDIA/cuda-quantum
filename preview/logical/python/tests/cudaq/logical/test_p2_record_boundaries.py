# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.logical as qlx


@qlx.gadget(implements=qlx.logical.idle)
def extraction_round(
    block: qlx.patch[qlx.codes.Steane],
    previous: qlx.types.record[qlx.codes.Steane],
) -> tuple[
        qlx.patch[qlx.codes.Steane],
        qlx.types.record[qlx.codes.Steane],
]:
    block, current = qlx.extract_syndrome(block)
    return block, current


@qlx.protocol(implements=qlx.logical.idle)
def two_rounds(
    block: qlx.patch[qlx.codes.Steane],
    initial: qlx.types.record[qlx.codes.Steane],
) -> tuple[
        qlx.patch[qlx.codes.Steane],
        qlx.types.record[qlx.codes.Steane],
]:
    block, current = extraction_round(block, initial)
    return extraction_round(block, current)


def test_syndrome_records_are_direct_typed_gadget_boundaries():
    text = qlx.compile(extraction_round).to_mlir()
    patch = ("!fabric.patch<@Steane, @Steane_default_encoding, "
             "@Steane_default_encoding_initial_epoch>")
    syndrome = ("!fabric.syndrome<@Steane, @Steane_default_encoding, "
                "@Steane_default_encoding_initial_epoch>")
    signature = (f"({patch}, {syndrome}) -> "
                 f"({patch}, {syndrome})")
    assert signature in text
    assert "fabric.read_syndrome_ancillas" in text
    assert "GadgetResult" not in text


def test_protocols_compose_record_boundaries():
    build = qlx.compile(two_rounds)
    text = build.to_mlir()
    assert text.count("fabric.call @extraction_round") == 2
    syndrome = ("!fabric.syndrome<@Steane, @Steane_default_encoding, "
                "@Steane_default_encoding_initial_epoch>")
    assert text.count(syndrome) >= 4
    assert build.module.operation.verify()

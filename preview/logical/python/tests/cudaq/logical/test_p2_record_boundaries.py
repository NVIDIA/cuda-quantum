# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.logical


@cudaq.logical.gadget(implements=cudaq.logical.std.idle)
def extraction_round(
    block: cudaq.logical.patch[cudaq.logical.codes.Steane],
    previous: cudaq.logical.types.record[cudaq.logical.codes.Steane],
) -> tuple[
        cudaq.logical.patch[cudaq.logical.codes.Steane],
        cudaq.logical.types.record[cudaq.logical.codes.Steane],
]:
    block, current = cudaq.logical.extract_syndrome(block)
    return block, current


@cudaq.logical.protocol(implements=cudaq.logical.std.idle)
def two_rounds(
    block: cudaq.logical.patch[cudaq.logical.codes.Steane],
    initial: cudaq.logical.types.record[cudaq.logical.codes.Steane],
) -> tuple[
        cudaq.logical.patch[cudaq.logical.codes.Steane],
        cudaq.logical.types.record[cudaq.logical.codes.Steane],
]:
    block, current = extraction_round(block, initial)
    return extraction_round(block, current)


def test_syndrome_records_are_direct_typed_gadget_boundaries():
    text = cudaq.logical.compile(extraction_round).to_mlir()
    patch = ("!fabric.patch<@Steane, @Steane_default_encoding, "
             "@Steane_default_encoding_initial_epoch>")
    syndrome = ("!fabric.syndrome<@Steane, @Steane_default_encoding, "
                "@Steane_default_encoding_initial_epoch>")
    signature = (f"({patch}, {syndrome}) -> "
                 f"({patch}, {syndrome})")
    assert signature in text
    assert "fabric.read_syndrome_ancillas" in text
    assert "GadgetResult" not in text


def test_protocols_compose_record_boundaries_without_analysis_annotations():
    build = cudaq.logical.compile(two_rounds)
    text = build.to_mlir()
    assert text.count("fabric.call @extraction_round") == 2
    syndrome = ("!fabric.syndrome<@Steane, @Steane_default_encoding, "
                "@Steane_default_encoding_initial_epoch>")
    assert text.count(syndrome) >= 4
    assert build.module.operation.verify()

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Structured gadget conditionals preserve private linear carry state."""

import cudaq.logical as cql

HIGH_RATE = cql.codes.CSSCode(
    name="high_rate_cond_test",
    n=2,
    k=2,
    d=1,
    block=cql.codes.CSSBlock(data=2),
    lx=((0,), (1,)),
    lz=((0,), (1,)),
)


def test_gadget_cond_preserves_bb_continuation_on_merged_patch():
    builder = cql.compiler.GadgetBuilder(
        "conditional_bb_continuation",
        implements=cql.logical.idle,
        signature={"block": cql.patch[HIGH_RATE]},
    )
    block = builder.input("block")
    block, condition = builder.mpp(cql.types.Z(block[0]))
    continuation = object()
    block._bb_syndrome_continuation = continuation

    def passthrough(branch_block):
        assert branch_block._bb_syndrome_continuation is continuation
        return (branch_block,)

    merged, = builder._backend.cond(
        condition,
        then=passthrough,
        else_=passthrough,
        carries=(block,),
    )

    assert merged._bb_syndrome_continuation is continuation
    # This sentinel models the private carry metadata only. Clear it before
    # returning because a real nonterminal BB cycle must be continued locally.
    merged._bb_syndrome_continuation = None
    build = builder.finish(merged)
    assert build.verify()

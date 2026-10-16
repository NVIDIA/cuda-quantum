# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Bounded-retry success semantics over ordinary returned gadget results."""

import cudaq.logical as cql

RetryCode = cql.codes.CSSCode(
    name="retry_success_profile_code",
    n=1,
    k=1,
    d=1,
    block=cql.codes.CSSBlock(data=1),
    lx=((0,),),
    lz=((0,),),
)


@cql.objective
def retry_measurement_objective(
    qubit: cql.types.logical_qubit,) -> tuple[cql.types.logical_qubit, bool]:
    return cql.mpp(cql.types.Z(qubit))


@cql.gadget(implements=retry_measurement_objective)
def retry_measurement(
    block: cql.patch[RetryCode],) -> tuple[cql.patch[RetryCode], bool]:
    block, result = cql.mpp(cql.types.Z(block[0]))
    return block, result


retry_profile = cql.gadgets.GadgetProfile(
    retry_measurement,
    success=(cql.gadgets.SuccessPredicate(
        cql.gadgets.ProfileParity.from_value(
            retry_measurement.record("mpp0.outcome"),) ^ True),),
    name="retry_measurement_success",
)


@cql.protocol
def retry_protocol(block: cql.patch[RetryCode],) -> cql.patch[RetryCode]:
    block, accepted = retry_measurement(block, analysis=retry_profile)
    return cql.ops.retry(block, until=accepted, max_attempts=4)


def test_profile_success_can_bind_an_inferred_result_row():
    build = cql.compile(retry_protocol)
    text = build.to_mlir()

    assert 'roles = [["result"]]' in text
    assert "fabric.success" in text
    assert "fabric.retry" in text
    assert build.module.operation.verify()

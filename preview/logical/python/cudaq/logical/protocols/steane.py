# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Concrete Steane resource-injection protocols.

These are ordinary linked Python definitions, not registry entries.  Their
bodies are the implementation selected for a Steane T-resource action and are
fully inspectable in Fabric/P3 IR.
"""

from __future__ import annotations

from .. import codes, std
from cudaq.logical.programs.decorators import objective
from cudaq.logical.ops._impl import (
    cond,
    cx,
    discard,
    measure_z,
    mz,
    parity,
    sdg,
    unpack_resource,
)
from cudaq.logical.types.values import logical_qubit
from cudaq.logical.gadgets import (
    gadget,
    patch,
)
from cudaq.logical.protocols.definition import protocol
from cudaq.logical.types.semantic import resource


@objective
def steane_teleportation_measurement_intent(
    compute: logical_qubit,
    magic: logical_qubit,
) -> tuple[logical_qubit, bool]:
    compute, magic = cx(compute, magic)
    return compute, measure_z(magic)


@gadget(implements=steane_teleportation_measurement_intent)
def steane_teleportation_measurement(
    compute: patch[codes.Steane],
    state: resource[std.T_STATE],
) -> tuple[patch[codes.Steane], bool]:
    # Transfer the resource payload into explicit encoded ownership. This is a
    # state handoff, not a logical operation.
    compute, magic = unpack_resource(state,
                                     like=compute,
                                     encoding=codes.BareQubit)

    # One-bit T gate teleportation: outcome zero already leaves T|psi>; outcome
    # one leaves T-dagger|psi> up to phase and therefore requires logical S.
    # The resource payload is a one-qubit encoded state. Coupling a logical-Z
    # representative of Steane to that carrier implements the logical CNOT
    # needed by gate teleportation without pretending that the two blocks have
    # the same physical width.
    compute, magic = cx(
        compute.data,
        magic.data,
        pairs=tuple((index, 0) for index in codes.Steane.lz[0]),
    )
    magic, raw_bits = mz(magic.data, record="magic_z")
    correction = parity(raw_bits)
    discard(magic)
    return compute, correction


@gadget(implements=std.s)
def steane_logical_s(block: patch[codes.Steane],) -> patch[codes.Steane]:
    # With the built-in Steane representatives, transversal S-dagger acts as
    # logical S: weight-four X stabilizers keep phase +1 and a weight-three
    # logical X maps to logical Y.
    return sdg(block.data)


@protocol(implements=std.t)
def steane_t_injection(
    compute: patch[codes.Steane],
    state: resource[std.T_STATE],
) -> patch[codes.Steane]:
    compute, correction = steane_teleportation_measurement(compute, state)
    compute, = cond(
        correction,
        carries=(compute,),
        then=lambda live: (steane_logical_s(live),),
        else_=lambda live: (live,),
    )
    return compute


__all__ = [
    "steane_teleportation_measurement_intent",
    "steane_teleportation_measurement",
    "steane_logical_s",
    "steane_t_injection",
]

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Derive and compose serial gadget models with occurrence-local noise."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from cudaq_edem import EDEM, compose_chain

from cudaq.logical.gadgets import GadgetContract
from cudaq.logical.compiler import (
    Depolarize1Placement,
    StandardDepolarizingNoise,
    compile,
    project_clifford,
)
from cudaq.logical.compiler.clifford import CliffordProjection
from .edem import FaultSet, _edem_from_build, standard_depolarizing_faults


def derive_edem_chain(
    profiles: Sequence[GadgetContract],
    *,
    noise: StandardDepolarizingNoise | None = None,
    depolarizations: Mapping[int, Sequence[Depolarize1Placement]] | None = None,
    fault_factory: Callable[[int, GadgetContract, CliffordProjection], FaultSet]
    | None = None,
) -> EDEM:
    """Derive a serial chain, reusing identical gadget/fault-set pairs.

    Contracts are supplied explicitly, for example through a recipe's
    ``gadget_contract(gadget)`` function. The result retains exposed code boundaries and
    all original fault groups; use ``cudaq_edem.project_DEM`` for a closed
    experiment. Repeated occurrences have distinct fault indicators.

    By default, use standard physical noise and per-occurrence depolarization
    placements, as in ``compiler.emit_stim``. Alternatively, ``fault_factory``
    receives (occurrence index, contract, Clifford projection) and
    returns the complete ``FaultSet`` for that occurrence. It cannot be combined
    with ``noise`` or ``depolarizations``. Probabilities remain part of each
    fault set; this helper does not introduce a separate distribution API.
    """
    profiles = tuple(profiles)
    if fault_factory is not None and (noise is not None or
                                      depolarizations is not None):
        raise ValueError("choose fault_factory or standard noise placements")
    if noise is None:
        noise = StandardDepolarizingNoise()
    placements = depolarizations or {}
    if any(i not in range(len(profiles)) for i in placements):
        raise ValueError("depolarization occurrence index is out of range")
    builds, models, occurrences = {}, {}, []
    for index, profile in enumerate(profiles):
        identity = id(profile)
        if identity not in builds:
            build = compile(profile)
            projection = project_clifford(build)
            builds[identity] = build, projection
        build, projection = builds[identity]
        faults = (
            fault_factory(index, profile, projection)
            if fault_factory is not None else standard_depolarizing_faults(
                projection, noise, depolarizations=placements.get(index, ())))
        key = identity, faults
        if key not in models:
            models[key] = _edem_from_build(build, faults=faults)
        occurrences.append(models[key])
    return compose_chain(occurrences)


__all__ = ["derive_edem_chain"]

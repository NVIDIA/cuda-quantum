# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q Logical machine recipes used by the built-in full-stack targets."""

from __future__ import annotations

from .. import codes, devices, gadgets


def _css_architecture(name: str, code) -> devices.QECArchitecture:
    """Build the standard generic CSS realization bundle for one code."""

    return devices.QECArchitecture(
        name,
        code.default_encoding,
        link_roots=(
            gadgets.prepare_zero(code),
            gadgets.prepare_plus(code),
            gadgets.css_memory_round(code),
            gadgets.measure_x(code),
            gadgets.measure_z(code),
            gadgets.logical_pauli(code, basis="x"),
            gadgets.logical_pauli(code, basis="z"),
        ),
    )


def surface_architecture(distance: int = 3) -> devices.QECArchitecture:
    """Return the explicitly linked surface-code realization bundle.

    The gadget factories capture their code in the patch type, so they must be
    created for the requested distance rather than retained as ``Surface[3]``
    module globals.
    """

    code = codes.Surface[distance]
    return _css_architecture(f"Surface{distance}Architecture", code)


def steane_architecture() -> devices.QECArchitecture:
    """Return the explicitly linked Steane-code realization bundle."""

    code = codes.Steane
    return _css_architecture("SteaneArchitecture", code)


__all__ = [
    "surface_architecture",
    "steane_architecture",
]

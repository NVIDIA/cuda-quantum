# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Steane-code QEC architecture recipe."""

from __future__ import annotations

from .. import codes, devices, gadgets

__all__ = ("architecture",)


def architecture() -> devices.QECArchitecture:
    """Return the explicitly linked Steane CSS realization bundle."""

    code = codes.Steane
    return devices.QECArchitecture(
        "SteaneArchitecture",
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

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable typed physical measurement definitions."""

from __future__ import annotations

from cudaq.logical.architecture.physical_definition import (
    QuantumProcess,
    PhysicalInstrument,
)

MZ = PhysicalInstrument(
    "measure_z_instrument",
    operation="measure",
    arity=1,
    record_schema="bit",
    preserves_inputs=True,
    process=QuantumProcess("measure_z"),
    controller_bindings={"lanes": "measure_z"},
)

MX = PhysicalInstrument(
    "measure_x_instrument",
    operation="measure_x",
    arity=1,
    record_schema="bit",
    preserves_inputs=True,
    process=QuantumProcess("measure_x"),
    controller_bindings={"lanes": "measure_x"},
)

MPP = PhysicalInstrument(
    "mpp",
    operation="measure_product",
    arity=None,
    record_schema="bit",
    preserves_inputs=True,
    process=QuantumProcess("measure_product"),
)

__all__ = ["PhysicalInstrument", "MZ", "MX", "MPP"]

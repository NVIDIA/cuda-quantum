# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Dynamic-code (Floquet-style) authoring library.

Binding this module links the typed artifacts a dynamic code
profile is written with: measurement phases with instantaneous stabilizer
groups, gauge-measurement maps, and record-defined logicals.

Concrete period geometries (honeycomb and friends) are research inputs, not
library built-ins yet: an example or package supplies the gauge families,
ISGs, and transfer maps, and these artifacts carry them through verification.
"""

from __future__ import annotations

from cudaq.logical.codes import (
    EncodingEpoch,
    EncodingEpochSchema,
    GaugeMeasurementMap,
    MeasurementPhase,
    RecordLogicalMap,
)

__all__ = [
    "MeasurementPhase",
    "EncodingEpoch",
    "EncodingEpochSchema",
    "GaugeMeasurementMap",
    "RecordLogicalMap",
]

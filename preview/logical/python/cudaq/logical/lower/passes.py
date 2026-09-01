# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from enum import Enum


class Pass(Enum):
    """Reserved CUDA-Q Logical pass vocabulary.

    CUDA-Q Logical pipelines use typed stages; research-only native pass spellings
    are deliberately absent from this enum.
    """


class Translation(Enum):
    FABRIC_TO_STIM = "fabric-to-stim"

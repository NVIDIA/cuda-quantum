# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Resource-transport vocabulary.

Binding this module links the typed transport surface used by production and
injection protocols: the P2
``cudaq.logical.produce``/``cudaq.logical.transport`` resource-flow
operations and the standard resource kinds they move. Encoded payload
adapters (``pack_resource``/``unpack_resource``) live on the ordinary op
surface; a device library with a concrete transport protocol exports it as a
normal ``@cudaq.logical.protocol`` implementing
``cudaq.logical.logical.transport(kind)``.
"""

from __future__ import annotations

from ..std import (  # noqa: F401  (re-exports)
    CCZ_STATE, RAW_T_STATE, T_STATE, produce, transport,
)

__all__ = ["T_STATE", "RAW_T_STATE", "CCZ_STATE", "produce", "transport"]

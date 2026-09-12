# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""qlx.dialects.fabric -- Fabric dialect Python module.

Re-exports TableGen-generated Fabric op/enum bindings and the
native extension's Fabric type/attribute classes.  Single-file
user-facing module, like ``mlir.dialects.arith`` upstream.
"""

from __future__ import annotations

# Generated op + enum bindings (declare_mlir_dialect_python_bindings).
from ._fabric_ops_gen import *  # noqa: F401,F403
from ._fabric_enum_gen import *  # noqa: F401,F403

# Native extension: Fabric type/attr subclasses live in the `fabric`
# submodule of the combined _qlx_ext extension.  The dialect itself is
# auto-registered on every Context via the _site_initialize_0 hook.
from .._mlir_libs._qlx_ext.fabric import (  # noqa: F401
    # Parameterized types
    PatchType, SyndromeType, ResourceStateType,
    # Simple types
    BitType, FrameType, SlotType, MachineType,
    # Enum attributes
    PartitionAttr, RoleAttr, PrepAttr, MergeBasisAttr, BoundaryAttr,
    ResourceAttr,
    # Composite attributes
    FloorplanAttr, FlowAttr, SpecOnlyAttr,
)

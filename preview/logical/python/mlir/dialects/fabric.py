# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""cudaq.mlir.dialects.fabric -- Fabric dialect Python module.

Re-exports TableGen-generated Fabric op/enum bindings and the
native extension's Fabric type/attribute classes.  Single-file
user-facing module, like ``mlir.dialects.arith`` upstream.
"""

from __future__ import annotations

# Generated bindings contain exactly the operations registered by the product
# dialect. Keep the public module explicit as a second, readable contract.
from ._fabric_ops_gen import (  # noqa: F401
    AllFalseOp, AllZeroOp, AllocOp, AssembleSyndromeOp, BarrierOp, CXOp, CZOp,
    CallOp, CircuitOp, CodeOp, CodeProfileOp, DeallocOp, DeviceOp,
    DiscardResourceOp, EncodingEpochOp, EncodingEpochSchemaOp,
    EncodingHierarchyOp, EncodingOp, EncodingPackOp, EncodingProjectionOp,
    EncodingUnpackOp, EpochTransitionOp, EstablishSupportOp, GadgetOp,
    GadgetSpecOp, HOp, IdleOp, IfOp, InitBasisOp, InjectOp, InterconnectOp,
    MapChildrenOp, MeasureBasisOp, MeasureGaugesOp, MeasureProductOp, MergeOp,
    MoveOp, MppOp, MultiMeasureOp, MzOp, ObjectiveOp, PackResourceOp, ParityOp,
    PatchGraphOp, PatchTransformOp, PermuteOp, PrepXOp, PrepZOp,
    ProduceResourceOp, ProtocolOp, ProtocolReturnOp, ReadSyndromeAncillasOp,
    RecvOp, RegionOp, RelocateOp, RepeatOp, ResetOp, ResourceRequestOp,
    ResourceRotateProductOp, RetryOp, ReturnOp, RotateProductOp, SOp, SdgOp,
    SelectionOp, SendOp, SplitOp, TOp, TdgOp, TransformBeginOp, TransformEndOp,
    TransportOp, TransversalCXOp, UnpackResourceOp, WhileConditionOp, WhileOp,
    XOp, XorOp, YieldOp, ZOp,
)
from ._fabric_enum_gen import (  # noqa: F401
    Boundary, Layout, MergeBasis, Partition, Prep, ResourceType, Role, Route,
)

# Native extension: Fabric type and attribute subclasses live in the `fabric`
# submodule of the combined _qlx_ext extension.  The dialect itself is
# auto-registered on every Context via the _site_initialize_1 hook.
from cudaq.mlir._mlir_libs._qlx_ext.fabric import (  # noqa: F401
    # Parameterized types
    PatchType, SyndromeType, ResourceStateType,
    # Simple types
    BitType, SlotType, MachineType,
    # Enum attributes
    PartitionAttr, RoleAttr, PrepAttr, MergeBasisAttr, BoundaryAttr,
    ResourceAttr,
    # Composite attributes
    FloorplanAttr, FlowAttr, SpecOnlyAttr,
)

__all__ = [
    "AllFalseOp",
    "AllZeroOp",
    "AllocOp",
    "AssembleSyndromeOp",
    "BarrierOp",
    "CXOp",
    "CZOp",
    "CallOp",
    "CircuitOp",
    "CodeOp",
    "CodeProfileOp",
    "DeallocOp",
    "DeviceOp",
    "DiscardResourceOp",
    "EncodingEpochOp",
    "EncodingEpochSchemaOp",
    "EncodingHierarchyOp",
    "EncodingOp",
    "EncodingPackOp",
    "EncodingProjectionOp",
    "EncodingUnpackOp",
    "EpochTransitionOp",
    "EstablishSupportOp",
    "GadgetOp",
    "GadgetSpecOp",
    "HOp",
    "IdleOp",
    "IfOp",
    "InitBasisOp",
    "InjectOp",
    "InterconnectOp",
    "MapChildrenOp",
    "MeasureBasisOp",
    "MeasureGaugesOp",
    "MeasureProductOp",
    "MergeOp",
    "MoveOp",
    "MppOp",
    "MultiMeasureOp",
    "MzOp",
    "ObjectiveOp",
    "PackResourceOp",
    "ParityOp",
    "PatchGraphOp",
    "PatchTransformOp",
    "PermuteOp",
    "PrepXOp",
    "PrepZOp",
    "ProduceResourceOp",
    "ProtocolOp",
    "ProtocolReturnOp",
    "ReadSyndromeAncillasOp",
    "RecvOp",
    "RegionOp",
    "RelocateOp",
    "RepeatOp",
    "ResetOp",
    "ResourceRequestOp",
    "ResourceRotateProductOp",
    "RetryOp",
    "ReturnOp",
    "RotateProductOp",
    "SOp",
    "SdgOp",
    "SelectionOp",
    "SendOp",
    "SplitOp",
    "TOp",
    "TdgOp",
    "TransformBeginOp",
    "TransformEndOp",
    "TransportOp",
    "TransversalCXOp",
    "UnpackResourceOp",
    "WhileConditionOp",
    "WhileOp",
    "XOp",
    "XorOp",
    "YieldOp",
    "ZOp",
    "Boundary",
    "Layout",
    "MergeBasis",
    "Partition",
    "Prep",
    "ResourceType",
    "Role",
    "Route",
    "PatchType",
    "SyndromeType",
    "ResourceStateType",
    "BitType",
    "SlotType",
    "MachineType",
    "PartitionAttr",
    "RoleAttr",
    "PrepAttr",
    "MergeBasisAttr",
    "BoundaryAttr",
    "ResourceAttr",
    "FloorplanAttr",
    "FlowAttr",
    "SpecOnlyAttr",
]

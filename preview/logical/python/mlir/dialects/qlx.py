# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Product CUDA-Q Logical dialect bindings for the registered P0-P2 surface."""

from __future__ import annotations

from ._qlx_ops_gen import (  # noqa: F401
    ActionOp, ApplyOp, CallOp, DeviceOp, DiscardOp, EstimateResultOp,
    ExperimentOp, IdleOp, IfOp, InstrumentDeclOp, InstrumentOp,
    LogicalToQECBindingOp, LoweringRecipeOp, MeasureOp, ObjectiveBodyOp,
    PrepareOp, ProgramOp, QECLoweringOp, RepeatOp, ReturnOp, SelectionOp,
    TargetManifestOp, WhileConditionOp, WhileOp, XorOp, YieldOp,
)
from ._qlx_enum_gen import BuiltinAction, BuiltinInstrument, Pauli  # noqa: F401
from cudaq.mlir._mlir_libs._qlx_ext.qlx import PauliAttr, set_inherent_attr  # noqa: F401

__all__ = [
    "ActionOp",
    "ApplyOp",
    "BuiltinAction",
    "BuiltinInstrument",
    "CallOp",
    "DeviceOp",
    "DiscardOp",
    "EstimateResultOp",
    "ExperimentOp",
    "IdleOp",
    "IfOp",
    "InstrumentDeclOp",
    "InstrumentOp",
    "LogicalToQECBindingOp",
    "LoweringRecipeOp",
    "MeasureOp",
    "ObjectiveBodyOp",
    "Pauli",
    "PauliAttr",
    "PrepareOp",
    "ProgramOp",
    "QECLoweringOp",
    "RepeatOp",
    "ReturnOp",
    "SelectionOp",
    "TargetManifestOp",
    "WhileConditionOp",
    "WhileOp",
    "XorOp",
    "YieldOp",
    "set_inherent_attr",
]

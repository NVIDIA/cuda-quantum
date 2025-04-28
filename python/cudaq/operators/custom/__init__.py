# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import typing
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.operators import *
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, MatrixOperatorTerm
from .custom_op import MatrixOperatorElement
from .backwards_compatibility import *

define = MatrixOperatorElement.define


def instantiate(op_id: str, degrees: int | typing.Iterable[int]):
    if isinstance(degrees, int):
        degrees = [degrees]
    element = MatrixOperatorElement(op_id, degrees)
    return MatrixOperatorTerm(element)

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import typing  # type: ignore

from .definitions import *  # for backwards compatibility
from .manipulation import OperatorArithmetics, _sum_transformation, _product_transformation, _evaluate

for op_type in typing.get_args(OperatorSum):
    op_type._transform = _sum_transformation
for op_type in typing.get_args(ProductOperator):
    op_type._transform = _product_transformation

for op_type in typing.get_args(OperatorSum):
    op_type.evaluate = _evaluate
for op_type in typing.get_args(ProductOperator):
    op_type.evaluate = _evaluate

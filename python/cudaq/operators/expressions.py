# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import typing  # type: ignore

from .definitions import * # for backwards compatibility
from .manipulation import OperatorArithmetics, _sum_evaluation, _product_evaluation, _evaluation

# FIXME: rename cppoperator to operatorsum etc

# FIXME: deprecate _evaluate or make public?
for op_type in typing.get_args(OperatorSum):
    op_type._evaluate = _sum_evaluation
for op_type in typing.get_args(ProductOperator):
    op_type._evaluate = _product_evaluation
for op_type in typing.get_args(ProductOperator):
    op_type.evaluate = _evaluation


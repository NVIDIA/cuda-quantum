# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, pytest, random
from cudaq.operator.scalar_op import ScalarOperator


def test_construction():
    def callback(x): return x
    scalar = ScalarOperator(callback)
    print(str(scalar))
    print(scalar.evaluate(x = 5))
    print(scalar.to_matrix(x = 5))


# for debugging
if __name__ == "__main__":
    pytest.main(["-rP"])
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
    scalar2 = ScalarOperator(lambda x: x*x)
    print(str(scalar))
    print(str(scalar2))
    print(scalar.evaluate(x = 5))
    print(scalar2.evaluate(x = 2))
    print(scalar.to_matrix(x = 5))
    print(scalar2.to_matrix(x = 2))
    assert ScalarOperator(5).is_constant()
    assert not scalar.is_constant()
    assert not scalar2.is_constant()
    assert 'x' in scalar2.parameters


# for debugging
if __name__ == "__main__":
    pytest.main(["-rP"])
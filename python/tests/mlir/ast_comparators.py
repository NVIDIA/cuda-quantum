# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os
import pytest
import cudaq


def test_comparison_operators_for_integers():

    @cudaq.kernel
    def test_integer_less_than():
        a = 3 < 5

    print(test_integer_less_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_greater_than():
        a = 5 > 3

    print(test_integer_greater_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_equal_to():
        a = 3 == 3

    print(test_integer_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_less_than_or_equal_to():
        a = 3 <= 5

    print(test_integer_less_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_greater_than_or_equal_to():
        a = 5 >= 5

    print(test_integer_greater_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_not_equal_to_true():
        a = 3 != 5

    print(test_integer_not_equal_to_true)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_not_equal_to_false():
        a = 5 != 5

    print(test_integer_not_equal_to_false)

    # CHECK-LABEL:    %false = arith.constant false


def test_comparison_operators_for_floats():

    @cudaq.kernel
    def test_float_less_than():
        a = 3.2 < 4.5

    print(test_float_less_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_greater_than():
        a = 4.5 > 3.2

    print(test_float_greater_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_equal_to():
        a = 3.2 == 3.2

    print(test_float_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_less_than_or_equal_to():
        a = 3.2 <= 4.5

    print(test_float_less_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_greater_than_or_equal_to():
        a = 4.5 >= 4.5

    print(test_float_greater_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_not_equal_to_true():
        a = 3.2 != 4.5

    print(test_float_not_equal_to_true)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_float_not_equal_to_false():
        a = 4.5 != 4.5

    print(test_float_not_equal_to_false)

    # CHECK-LABEL:    %false = arith.constant false


def test_comparison_operators_for_mixed_types():

    @cudaq.kernel
    def test_mixed_less_than():
        a = 3 < 4.5

    print(test_mixed_less_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_mixed_greater_than():
        a = 4.5 > 3

    print(test_mixed_greater_than)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_mixed_equal_to():
        a = 3 == 3.0

    print(test_mixed_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_mixed_less_than_or_equal_to():
        a = 3 <= 4.5

    print(test_mixed_less_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_mixed_greater_than_or_equal_to():
        a = 4.5 >= 4

    print(test_mixed_greater_than_or_equal_to)

    # CHECK-LABEL:    %true = arith.constant true

    @cudaq.kernel
    def test_mixed_not_equal_to_true():
        a = 3 != 4.5

    print(test_mixed_not_equal_to_true)

    # CHECK-LABEL:    %true = arith.constant true


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])

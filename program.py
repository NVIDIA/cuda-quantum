# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_param_negative_int_list():

    @cudaq.kernel
    def kernel(l: list[int], i: int) -> int:
        return l[i]

    lst = [0, 1, -1]
    for i in range(len(lst)):
        assert kernel(lst, i) == lst[i]

test_param_negative_int_list()
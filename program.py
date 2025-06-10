# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_return_bool():

    @cudaq.kernel
    def simple_bool_no_args() -> bool:
        return True

    results = cudaq.run_async(simple_bool_no_args, shots_count=2).get()
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True

test_return_bool()
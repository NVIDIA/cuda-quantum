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
        qubits = cudaq.qvector(2)
        return True

    # TODO: seg fault on running any kernel with no args
    results = cudaq.run(simple_bool_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True

test_return_bool()
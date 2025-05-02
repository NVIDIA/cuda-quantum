# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_no_return():

    @cudaq.kernel
    def simple_bool_no_args() -> bool:
        qubits = cudaq.qvector(2)
        return True

    results = cudaq.sample(simple_bool_no_args, shots_count=2)
    print(simple_bool_no_args)
    print(results)

test_no_return()

def test_return_bool():

    @cudaq.kernel
    def simple_bool_no_args() -> bool:
        #qubits = cudaq.qvector(2)
        return True

    print(simple_bool_no_args)
    # TODO: seg fault on running any kernel with no args
    results = cudaq.run(simple_bool_no_args, shots_count=2)
    
    print(results)

test_return_bool()
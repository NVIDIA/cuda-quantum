# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from typing import Tuple
import numpy as np

def test_return_list():

    @cudaq.kernel
    def simple_list_bool() -> list[bool]:
        qubits = cudaq.qvector(2)
        result = [True, False]
        return result
    
    print(cudaq.run(simple_list_bool, shots_count=2))

    @cudaq.kernel
    def simple_list_int() -> list[int]:
        qubits = cudaq.qvector(2)
        result = [2, 0]
        return result
    
    print(cudaq.run(simple_list_int, shots_count=2))

    # @cudaq.kernel
    # def simple_list_int32() -> list[np.int32]:
    #     qubits = cudaq.qvector(2)
    #     result = [1, 0]
    #     return result
    
    # print(cudaq.run(simple_list_int32, shots_count=2))

    @cudaq.kernel
    def simple_list_int64() -> list[np.int64]:
        qubits = cudaq.qvector(2)
        result = [3, 0]
        return result
    
    print(cudaq.run(simple_list_int64, shots_count=2))

    @cudaq.kernel
    def simple_list_float() -> list[float]:
        qubits = cudaq.qvector(2)
        result = [4.0, 0.0]
        return result
    
    print(cudaq.run(simple_list_float, shots_count=2))

    @cudaq.kernel
    def simple_list_float32() -> list[np.float32]:
        qubits = cudaq.qvector(2)
        result = [1.0, 0.0]
        return result

    print(cudaq.run(simple_list_float32, shots_count=2))

    @cudaq.kernel
    def simple_list_float64() -> list[np.float64]:
        qubits = cudaq.qvector(2)
        result = [6.0, 0.0]
        return result
    
    print(cudaq.run(simple_list_float64, shots_count=2))


test_return_list()


# def test_return_tuple():

#     @cudaq.kernel
#     def simple_list_tuple() -> Tuple[int, bool]:
#         qubits = cudaq.qvector(2)
#         result = [13, True]
#         return result
    
#     print(cudaq.run(simple_list_tuple, 2, shots_count=2))

# test_return_tuple()
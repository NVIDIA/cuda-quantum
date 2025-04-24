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
import re
from dataclasses import dataclass

def test_return_list():

    @cudaq.kernel
    def simple_list_bool() -> list[bool]:
        qubits = cudaq.qvector(2)
        result = [True, False]
        return result
    
    print("simple_list_bool:")
    print(cudaq.run(simple_list_bool, shots_count=2))

    @cudaq.kernel
    def simple_list_int() -> list[int]:
        qubits = cudaq.qvector(2)
        result = [2, 0]
        return result
    
    print("simple_list_int:")
    print(cudaq.run(simple_list_int, shots_count=2))

    @cudaq.kernel
    def simple_list_int32() -> list[np.int32]:
        qubits = cudaq.qvector(2)
        result = [1, 0]
        return result
    
    print("simple_list_int32:")
    print(cudaq.run(simple_list_int32, shots_count=2))

    @cudaq.kernel
    def simple_list_int64() -> list[np.int64]:
        qubits = cudaq.qvector(2)
        result = [3, 0]
        return result
    
    print("simple_list_int64:")
    print(cudaq.run(simple_list_int64, shots_count=2))

    @cudaq.kernel
    def simple_list_float() -> list[float]:
        qubits = cudaq.qvector(2)
        result = [4.0, 0.0]
        return result
    
    print("simple_list_float:")
    print(cudaq.run(simple_list_float, shots_count=2))

    @cudaq.kernel
    def simple_list_float32() -> list[np.float32]:
        qubits = cudaq.qvector(2)
        result = [1.0, 0.0]
        return result

    print("simple_list_float32:")
    print(cudaq.run(simple_list_float32, shots_count=2))

    @cudaq.kernel
    def simple_list_float64() -> list[np.float64]:
        qubits = cudaq.qvector(2)
        result = [6.0, 0.0]
        return result
    
    print("simple_list_float64:")
    print(cudaq.run(simple_list_float64, shots_count=2))

test_return_list()


def test_return_tuple():

    @cudaq.kernel
    def simple_tuple_int_float(n: int, t: tuple[int, float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        return t

    print(cudaq.run(simple_tuple_int_float, 2, (13, 42.3), shots_count=2))

    @cudaq.kernel
    def simple_tuple_float_int(n: int, t: tuple[float, int]) -> tuple[float, int]:
        qubits = cudaq.qvector(n)
        return t

    print(cudaq.run(simple_tuple_float_int, 2, (42.3, 13), shots_count=2))

    @cudaq.kernel
    def simple_tuple_bool_int(n: int, t: tuple[bool, int]) -> tuple[bool, int]:
        qubits = cudaq.qvector(n)
        return t
    # TODO: fix alignment?
    print(cudaq.run(simple_tuple_bool_int, 2, (True, 13), shots_count=2))

    @cudaq.kernel
    def simple_tuple_int_bool(n: int, t: tuple[int, bool]) -> tuple[int, bool]:
        qubits = cudaq.qvector(n)
        return t

    print(cudaq.run(simple_tuple_int_bool, 2, (13, True), shots_count=2))

    @cudaq.kernel
    def simple_tuple_bool_int_float(n: int, t: tuple[bool, int, float]) -> tuple[bool, int, float]:
        qubits = cudaq.qvector(n)
        return t
    # TODO: fix alignment?
    print(cudaq.run(simple_tuple_bool_int_float, 2, (True, 13, 42.3), shots_count=2))

test_return_tuple()

# def test_return_Tuple():

#     @cudaq.kernel
#     def simple_tuple2(n: int, t: Tuple[int, float]) -> Tuple[int, float]:
#         qubits = cudaq.qvector(n)
#         return t
    
#     print(cudaq.run(simple_tuple2, 2, (14, 42.3), shots_count=2))

# test_return_Tuple()

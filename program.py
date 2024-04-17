# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, sys

import pytest
import numpy as np
import math
from typing import List 

import cudaq
from cudaq import spin

def test_call():

    cudaq.reset_target()

    @cudaq.kernel(verbose=True)
    def test_param(i: int, v1: list[int]) -> int: 
        return i

    l = [42]
    print(test_param(0, l))
    print(test_param(1, l))
    print(test_param(2, l))
    print(test_param(3, l))

test_call() 

def test_python_call():

    def test_param(i: int, v1: list[int]) -> int: 
        return i

    l = [42]
    print(test_param(0, l))
    print(test_param(1, l))
    print(test_param(2, l))
    print(test_param(3, l))

test_python_call()
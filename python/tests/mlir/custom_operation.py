# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest
import numpy as np
import cudaq


def test_bell_pair():

    cudaq.register_operation("custom_h", 1, 0, 
                                        1. / np.sqrt(2.) *
                                        np.array([[1, 1], [1, -1]]))
    cudaq.register_operation("custom_x", 1, 0, 
                                        np.array([[0, 1], [1, 0]]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    print(bell)

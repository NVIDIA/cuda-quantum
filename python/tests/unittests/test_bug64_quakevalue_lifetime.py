# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest

import cudaq

def test_QuakeValueLifetimeAndPrint(): 
    circuit = cudaq.make_kernel()
    qubitRegister = circuit.qalloc(2)
    circuit.x(qubitRegister[0])  
    s = str(circuit)
    print(s)

    assert s.count('quake.x') == 1

    circuit.x(qubitRegister[0])
    s = str(circuit)
    print(s)
    assert s.count('quake.x') == 2

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])

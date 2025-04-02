# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def test_quantinuum_capture():
    state = cudaq.State.from_data(np.array([1, 0], dtype = np.complex128))
    
    @cudaq.kernel()
    def kernel():
        q = cudaq.qvector(state)
    
    counts = cudaq.sample(kernel)
    print(counts)
    state = cudaq.State.from_data(np.array([1, 0, 0 ,0], dtype = np.complex128))
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)
   
    counts = cudaq.sample(kernel)
    print(counts)
    # assert '10' in counts
    # assert len(counts) == 1

# print('nvidia')
# cudaq.set_target('nvidia', option='fp64')
# test_quantinuum_capture()

# print('quantinuum')
# cudaq.set_target('quantinuum', emulate=True)
# test_quantinuum_capture()

def test_quantinuum():
    
    @cudaq.kernel()
    def kernel():
        q = cudaq.qvector(2)
    
    @cudaq.kernel()
    def kernel2(s: cudaq.State):
        q = cudaq.qvector(s)

    state = cudaq.get_state(kernel)
    counts = cudaq.sample(kernel2, state)
    print(counts)

print('quantinuum')
cudaq.set_target('quantinuum', emulate=True)
test_quantinuum()
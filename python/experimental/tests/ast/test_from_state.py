# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

# FIXME implement in eager mode...

@pytest.fixture(autouse=True)
def do_something():
    cudaq.__clearKernelRegistries()
    yield 
    return 

def test_from_state():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        cudaq.initialize_state(q, np.array([1./np.sqrt(2),0,0,1./np.sqrt(2)]))

    print(bell)
    counts = cudaq.sample(bell)
    counts.dump()
    assert '00' in counts and '11' in counts 


def test_from_state_again2():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        vector = np.array([1./np.sqrt(2),0,0,1./np.sqrt(2)])
        cudaq.initialize_state(q, vector)

    print(bell)
    counts = cudaq.sample(bell)
    counts.dump()
    assert '00' in counts and '11' in counts 

def test_from_state_again4():
    @cudaq.kernel(jit=True)
    def bell(vector: list[complex] ):
        q = cudaq.qvector(2)
        cudaq.initialize_state(q, vector)

    print(bell)

    # can pass a numpy array of complex
    vec = np.asarray([1./np.sqrt(2),0,0,1./np.sqrt(2)],dtype=complex)
    counts = cudaq.sample(bell, vec)
    counts.dump()
    assert '00' in counts and '11' in counts 

    # Make sure we throw an error if you pass wrong typed array
    with pytest.raises(RuntimeError) as error:
        vec = [1./np.sqrt(2),0,0,1./np.sqrt(2)]
        cudaq.sample(bell, vec)
    
    # Can pass a list of complex
    vec = [1./np.sqrt(2) + 0j,0j,0j,1./np.sqrt(2)+0j]
    cudaq.sample(bell, vec)

    # Another test
    vec = 1./np.sqrt(2) * np.asarray([1, 0, 0, 1], dtype=complex)
    cudaq.sample(bell, vec)
    

# WILL TAKE SOME WORK
# def test_from_state_again3():
#     @cudaq.kernel(jit=True)
#     def bell():
#         q = cudaq.qvector(2)
#         vector = 1/np.sqrt(2) * np.array([1,0,0,1])
#         cudaq.initialize_state(q, vector)

#     print(bell)
#     counts = cudaq.sample(bell)
#     counts.dump()
#     assert '00' in counts and '11' in counts 

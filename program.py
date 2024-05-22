# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def test_state():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    # Get the quantum state, which should be a vector.
    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])

    state = cudaq.get_state(bell)
    state.dump()

    #c = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]
    #expectedState = np.array(c, dtype=float)
    print(f"!!!State: {state}")
    #assert np.allclose(state, np.array(expectedState))

    @cudaq.kernel(verbose=True)
    def kernel(initialState: cudaq.State):
        qubits = cudaq.qvector(initialState)
    
    state = cudaq.get_state(kernel, state)
    # counts = cudaq.sample(kernel, state)
    print(state)
    # assert '11' in counts
    # assert '00' in counts
    # assert not '10' in counts
    # assert not '01' in counts

#test_state()

def test_state():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]
    state = cudaq.State.from_data(np.array(c, dtype=np.complex128))
    print(f"!!!State: {state}")

    @cudaq.kernel(verbose=True)
    def kernel(initialState: cudaq.State):
        qubits = cudaq.qvector(initialState)
    
    state2 = cudaq.get_state(kernel, state)
    print(state2)

test_state()

def test_state():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]
    state = np.array(c, dtype=np.complex128)
    print(f"!!!State: {state}")

    @cudaq.kernel(verbose=True)
    def kernel():
        qubits = cudaq.qvector(state)
    
    state2 = cudaq.get_state(kernel)
    print(state2)

#test_state()
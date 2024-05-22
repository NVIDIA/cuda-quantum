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

    @cudaq.kernel(verbose=True)
    def kernel(initialState: cudaq.State):
        qubits = cudaq.qvector(initialState)
    
    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts
    assert not '10' in counts
    assert not '01' in counts

test_state()


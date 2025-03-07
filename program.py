# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
from typing import Callable
import numpy as np

import cupy as cp

#cudaq.set_target('quantinuum', emulate=True)
# #cudaq.set_target("remote-mqpu", auto_launch="1")

# @cudaq.kernel
# def init():
#     q = cudaq.qvector(2)

# @cudaq.kernel
# def kernel(s: cudaq.State):
#     q = cudaq.qvector(s)


# state = cudaq.get_state(init)
# counts = cudaq.sample(kernel, state)
# print(counts)

#cudaq.set_target("quantinuum", emulate = True)
#cudaq.set_target("remote-mqpu", auto_launch="1")

# def test_grover():
#     """Test that compute_action works in tandem with kernel composability."""

#     @cudaq.kernel
#     def reflect(qubits: cudaq.qview):
#         ctrls = qubits.front(qubits.size() - 1)
#         last = qubits.back()
#         cudaq.compute_action(lambda: (h(qubits), x(qubits)),
#                              lambda: z.ctrl(ctrls, last))

#     @cudaq.kernel
#     def oracle(q: cudaq.qview):
#         z.ctrl(q[0], q[2])
#         z.ctrl(q[1], q[2])

#     print(reflect)

#     @cudaq.kernel
#     def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
#         q = cudaq.qvector(N)
#         h(q)
#         for i in range(M):
#             oracle(q)
#             reflect(q)
#         mz(q)

#     print(grover)
#     print(oracle)

#     counts = cudaq.sample(grover, 3, 1, oracle)
#     print(counts)
#     #assert len(counts) == 2
#     # assert '101' in counts
#     # assert '011' in counts

# test_grover()

#cudaq.set_target("quantinuum", emulate = True)
# cudaq.set_target("remote-mqpu", auto_launch="1")


# def test_state_vector_simple_py_float():
#     # Test overlap with device state vector
#     kernel = cudaq.make_kernel()
#     q = kernel.qalloc(2)
#     kernel.h(q[0])
#     kernel.cx(q[0], q[1])

#     # State is on the GPU, this is nvidia target
#     state = cudaq.get_state(kernel)
#     # Create a state on GPU
#     expected = cp.array([.707107, 0, 0, .707107])

#     # We are using the nvidia-fp32 target by default, it requires
#     # cupy overlaps to also have complex f32 data types, this
#     # should throw since we are using float data types only
#     with pytest.raises(RuntimeError) as error:
#         result = state.overlap(expected)

# test_state_vector_simple_py_float()

# def test_state_vector_simple_cfp32():
#     # Test overlap with device state vector
#     kernel = cudaq.make_kernel()
#     q = kernel.qalloc(2)
#     kernel.h(q[0])
#     kernel.cx(q[0], q[1])

#     # State is on the GPU, this is nvidia target
#     state = cudaq.get_state(kernel)
#     state.dump()
#     # Create a state on GPU
#     expected = cp.array([.707107, 0, 0, .707107], dtype=cp.complex64)
#     # Compute the overlap
#     result = state.overlap(expected)
#     assert np.isclose(result, 1.0, atol=1e-3)

# test_state_vector_simple_cfp32()

def test_ctrl_rotation_integration():
    """
    Tests more complex controlled rotation kernels, including
    pieces that will only run in quantinuum emulation.
    """
    cudaq.set_random_seed(4)
    cudaq.set_target("quantinuum", emulate=True)
    cudaq.set_random_seed(4)

    kernel = cudaq.make_kernel()
    ctrls = kernel.qalloc(4)
    ctrl = kernel.qalloc()
    target = kernel.qalloc()

    # Subset of `ctrls` in |1> state.
    kernel.x(ctrls[0])
    kernel.x(ctrls[1])

    # Multi-controlled rotation with that qreg should have
    # no impact on our target, since not all `ctrls` are |1>.
    kernel.cry(1.0, ctrls, target)

    # Flip the rest of our `ctrls` to |1>.
    kernel.x(ctrls[2])
    kernel.x(ctrls[3])

    # Multi-controlled rotation should now flip our target.
    kernel.crx(np.pi / 4., ctrls, target)

    # Test (1) (only works in emulation): mixed list of veqs and qubits.
    # Has no impact because `ctrl` = |0>
    kernel.crx(1.0, [ctrls, ctrl], target)
    # Test (2): Flip `ctrl` and try again.
    kernel.x(ctrl)
    kernel.crx(np.pi / 4., [ctrls, ctrl], target)

    result = cudaq.sample(kernel)
    print(result)

    # The `target` should be in a 50/50 mix between |0> and |1>.
    extra_mapping_qubits = "0000"
    want_1_state = extra_mapping_qubits + "111111"
    want_0_state = extra_mapping_qubits + "111110"
    assert result[want_1_state] == 505
    assert result[want_0_state] == 495
    cudaq.reset_target()

test_ctrl_rotation_integration()
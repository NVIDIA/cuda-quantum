# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

# def is_close(expected, actual):
#     return np.isclose(expected, actual, atol=1e-6)

# def test_kernel_simulation_dtype_complex_params():
#     cudaq.reset_target()
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))
    
#     cudaq.sample(kernel, c).dump()

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         ar = np.array(vec, dtype=cudaq.complex())
#         q = cudaq.qvector(ar)

#     cudaq.sample(kernel, c).dump()

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         v = vec
#         ar = np.array(v, dtype=cudaq.complex())
#         q = cudaq.qvector(ar)

#     cudaq.sample(kernel, c).dump()

# # test_kernel_simulation_dtype_complex_params()

# def test_kernel_simulation_dtype_complex_params_128():
#     cudaq.reset_target()
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         q = cudaq.qvector(np.array(vec, dtype=np.complex128))

#     cudaq.sample(kernel, c).dump()

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         ar = np.array(vec, dtype=np.complex128)
#         q = cudaq.qvector(ar)

#     cudaq.sample(kernel, c).dump()

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         v = vec
#         ar = np.array(v, dtype=np.complex128)
#         q = cudaq.qvector(ar)

#     cudaq.sample(kernel, c).dump()

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         v = vec
#         ar = np.array(v, dtype=np.complex128)
#         q = cudaq.qvector(ar)

#     cudaq.sample(kernel, c).dump()

# test_kernel_simulation_dtype_complex_params_128()

# def test_np_complex128_definition():
#     """Test that we can define complex lists inside kernel functions."""

#     # Define a list of complex inside a kernel
#     c = [
#         np.complex128(.70710678 + 1j),
#         np.complex128(0. + 2j),
#         np.complex128(0.),
#         np.complex128(0.70710678)
#     ]

#     @cudaq.kernel
#     def complex_vec_definition_real(i: int) -> float:
#         v = [
#             np.complex128(.70710678 + 1j),
#             np.complex128(0. + 2j),
#             np.complex128(0.),
#             np.complex128(0.70710678)
#         ][i]
#         return v.real

#     for i in range(len(c)):
#         assert is_close(c[i].real, complex_vec_definition_real(i))

# test_np_complex128_definition()

# def test_kernel_dtype_complex_complex128_params_f64():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]
#     arr = np.array(c, dtype=complex)

#     @cudaq.kernel
#     def kernel(vec : np.ndarray):
#         q = cudaq.qvector(np.array(vec, dtype=np.complex128))

#     kernel(arr)

# test_kernel_dtype_complex_complex128_params_f64()



# cudaq.reset_target()
# cudaq.set_target('nvidia')

# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # copy and cast the vector elements to complex64
#     q = cudaq.qvector(np.array(vec, dtype=np.complex64)) 

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }



# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # runtime error, simulation precision error
#     q = cudaq.qvector(np.array(vec, dtype=np.complex128))

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }


# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # copy and cast the vector elements to simulation data type
#     q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

# state = np.array([.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

# state = cudaq.State([.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

# @cudaq.kernel
# def kernel():
#     q = cudaq.qvector(state)

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }


# ## no arrays

# cudaq.reset_target()
# cudaq.set_target('nvidia')

# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # copy and cast vector elements to complex64 on f32 only
#     q = cudaq.qvector(vec) 

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }



# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # runtime error, simulation precision error
#     q = cudaq.qvector(np.array(vec, dtype=np.complex128))

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }


# c = [.70710678 + 0j, 0., 0., 0.70710678]

# @cudaq.kernel
# def kernel(vec: list[complex]):
#     # copy and cast the vector elements to simulation data type
#     q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

# counts = cudaq.sample(kernel)
# print(counts) # { 11:506 00:494 }

def test_builder_params():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    v = [0., 0., 0., 1.]

    kernel, state = cudaq.make_kernel(list[complex])
    q = kernel.qalloc(state)

    # Can now operate on the qvector as usual:
    # Rotate state of the front qubit 180 degrees along X.
    kernel.x(q[0])
    # Rotate state of the back qubit 180 degrees along Y.
    kernel.y(q[1])
    # Put qubits into superposition state.
    kernel.h(q)

    # Measure.
    kernel.mz(q)
    print(kernel)

    cudaq.sample(kernel, v).dump()

# test_builder_params()

def test_kernel_complex_params_rotate():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [0.+0j, 0., 0., 1.]


    # Our kernel will start with 2 qubits in `11`, then
    # rotate each qubit back to `0` before applying a
    # Hadamard gate.
    @cudaq.kernel(verbose=True)
    def kernel(vec : list[complex]):
        q = cudaq.qvector(vec)
        # Can now operate on the qvector as usual:
        # Rotate state of the front qubit 180 degrees along X.
        x(q.front())
        # Rotate state of the back qubit 180 degrees along Y.
        y(q.back())
        # Put qubits into superposition state.
        h(q)

        # Measure.
        mz(q)

    # TODO: error: 'quake.alloca' op init_state must be the only use
    cudaq.sample(kernel, c).dump()

#test_kernel_complex_params_rotate()

def test_precision_f64():
    cudaq.set_target('nvidia-fp64')
    # TODO: wrong simulator precision
    state = np.array([.70710678, 0., 0., 0.70710678], dtype=np.complex64)

    @cudaq.kernel(verbose=True)
    def kernel():
        # should convert
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

# test_precision_f64()

def test_alloca():
    cudaq.set_target('nvidia-fp64')
    # error: 'quake.alloca' op size operand required
    @cudaq.kernel(verbose=True)
    def kernel():
        qubits = cudaq.qvector(np.array([1., 0., 0.], dtype=complex))

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel)


test_alloca()

def test_alloca_builder():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')
    
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(np.array([1., 0., 0.], dtype=complex))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

test_alloca_builder()
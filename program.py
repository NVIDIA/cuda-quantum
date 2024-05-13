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



cudaq.reset_target()
cudaq.set_target('nvidia')

c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # copy and cast the vector elements to complex64
    q = cudaq.qvector(np.array(vec, dtype=np.complex64)) 

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }



c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # runtime error, simulation precision error
    q = cudaq.qvector(np.array(vec, dtype=np.complex128))

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }


c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # copy and cast the vector elements to simulation data type
    q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

state = np.array([.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

state = cudaq.State([.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

@cudaq.kernel
def kernel():
    q = cudaq.qvector(state)

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }


## no arrays

cudaq.reset_target()
cudaq.set_target('nvidia')

c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # copy and cast vector elements to complex64 on f32 only
    q = cudaq.qvector(vec) 

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }



c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # runtime error, simulation precision error
    q = cudaq.qvector(np.array(vec, dtype=np.complex128))

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }


c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel(vec: list[complex]):
    # copy and cast the vector elements to simulation data type
    q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

counts = cudaq.sample(kernel)
print(counts) # { 11:506 00:494 }
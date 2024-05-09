# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

# #fails
# def test_builder():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     v = [0., 1., 1., 0.]

#     kernel = cudaq.make_kernel()
#     qubits = kernel.qalloc(v)

#     cudaq.sample(kernel).dump()


#works
def test_builder_params():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    #v = [0., 1., 1., 0.]
    v = [.70710678, 0., 0., 0.70710678]

    kernel, state = cudaq.make_kernel(list[complex])
    qubits = kernel.qalloc(state)

    cudaq.sample(kernel, v).dump()


# # works but gives strange results
# def test_builder_params_float():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     # Note:
#     # Result: { 10:504 00:496 }
#     # should it be { 01:504 10:496 }?
#     v = [0., 1., 1., 0.]
#     kernel, state = cudaq.make_kernel(list[float])
#     qubits = kernel.qalloc(state)

#     cudaq.sample(kernel, v).dump()

# ##########################
# # Test creating a kernel
# ##########################

# vector of float

# def test_kernel_float_params():

#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel(vec : list[float]):
#         q = cudaq.qvector(vec)

#     cudaq.sample(kernel, f).dump()

# def test_kernel_float_capture():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel():
#         q = cudaq.qvector(f)

#     cudaq.sample(kernel).dump()

# def test_kernel_float_np_array_from_capture():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel():
#         q = cudaq.qvector(np.array(f))

#     cudaq.sample(kernel).dump()

# vector of complex

# def test_kernel_complex_params_rotate():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]

#     @cudaq.kernel
#     def kernel(vec : list[complex]):
#         q = cudaq.qvector(vec)
#         # Can now operate on the qvector as usual:
#         # Rotate state of the front qubit 180 degrees along X.
#         x(q.front())
#         # Rotate state of the back qubit 180 degrees along Y.
#         y(q.back())
#         # Put qubits into superposition state.
#         h(q)

#         # Measure.
#         mz(q)

#     # TODO: error: 'quake.alloca' op init_state must be the only use
#     cudaq.sample(kernel, c).dump()


def test_kernel_complex_params():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]
    #c = [0.+0j, 0., 0., 1.]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)

    cudaq.sample(kernel, c).dump()


def test_kernel_complex_capture():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(c)

    cudaq.sample(kernel).dump()


def test_kernel_complex_np_array_from_capture():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(np.array(c))

    cudaq.sample(kernel).dump()


def test_kernel_complex_definition():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1.0 + 0j, 0., 0., 1.])

    cudaq.sample(kernel).dump()


# np arrays with various dtypes


def test_kernel_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=complex))

    cudaq.sample(kernel, c).dump()


def test_kernel_dtype_complex128_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex128))

    cudaq.sample(kernel, c).dump()


def test_kernel_dtype_complex64_params_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex64))

    cudaq.sample(kernel, c).dump()


# def test_kernel_dtype_complex_complex128_params_f64():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]
#     arr = np.array(c, dtype=complex)
#     print(arr.data)

#     @cudaq.kernel
#     def kernel(vec : np.ndarray):
#         q = cudaq.qvector(np.array(vec, dtype=np.complex128))

#     # Needs https://github.com/NVIDIA/cuda-quantum/pull/1610
#     # RuntimeError: error: Invalid runtime argument type. Argument of type list[complex] was provided, but list[float] was expected.
#     cudaq.sample(kernel, arr).dump()

# simulation scalar


def test_kernel_simulation_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    cudaq.sample(kernel, c).dump()


def test_kernel_simulation_dtype_complex_params_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    cudaq.sample(kernel, c).dump()


# def test_kernel_simulation_dtype_np_array_params():
#     cudaq.reset_target()

#     c = [.70710678 + 0j, 0., 0., 0.70710678]

#     @cudaq.kernel(verbose=True)
#     def kernel(vec : np.array):
#         q = cudaq.qvector(vec)

#     # TODO 1: cudaq.kernel.ast_bridge.CompilerError: test_kernel_state_init.py:198: error: np.array is not a supported type.
#     cudaq.sample(kernel, np.array(c, dtype=cudaq.complex())).dump()


def test_kernel_simulation_dtype_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def test_scalar_vec7():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    cudaq.sample(test_scalar_vec7).dump()


def test_kernel_simulation_dtype_np_array_from_capture_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def test_scalar_vec7():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    cudaq.sample(test_scalar_vec7).dump()


def test_kernel_simulation_dtype_np_array_capture():
    cudaq.reset_target()
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    state = np.array(c, dtype=cudaq.complex())

    @cudaq.kernel(verbose=True)
    def test_scalar_vec8():
        q = cudaq.qvector(state)

    cudaq.sample(test_scalar_vec8).dump()

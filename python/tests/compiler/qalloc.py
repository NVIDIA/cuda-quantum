# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq


def test_kernel_qalloc_empty():
    """
    Test `cudaq.Kernel.qalloc()` when no arguments are provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with no function arguments.
    qubit = kernel.qalloc()
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qreg():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a register of
    qubits is provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 10 qubits allocated.
    qubit = kernel.qalloc(10)
    # Assert that 10 qubits have been allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<10>
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qreg_keyword():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a register of
    qubits is provided with a keyword argument.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 10 qubits allocated.
    qubit = kernel.qalloc(qubit_count=10)
    # Assert that 10 qubits have been allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<10>
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_quake_val():
    """
    Test `cudaq.Kernel.qalloc()` when a `QuakeValue` is provided.
    """
    kernel, value = cudaq.make_kernel(int)
    qreg = kernel.qalloc(value)
    qubit_count = 10
    kernel(qubit_count)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<?>[%[[VAL_0]] : i32]
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qubit():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a single qubit
    is provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 1 qubit allocated.
    qubit = kernel.qalloc(1)
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qubit_keyword():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a single qubit
    is provided with a keyword argument.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 1 qubit allocated and `qubit_count` keyword used.
    qubit = kernel.qalloc(qubit_count=1)
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])

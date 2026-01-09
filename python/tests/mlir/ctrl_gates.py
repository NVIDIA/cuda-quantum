# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

import cudaq


def test_kernel_2q_ctrl():
    """
    Test the `cudaq.Kernel` on each two-qubit gate (controlled 
    single qubit gates). We alternate the order of the control and target
    qubits between each successive gate.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # First three gates check the overload for providing a single control
    # qubit as a list of length 1.
    # Test both with and without keyword arguments.
    kernel.ch([qubit_0], qubit_1)
    kernel.cx([qubit_1], qubit_0)
    kernel.cy([qubit_0], qubit_1)
    # Check the overload for providing a single control qubit on its own.
    # Test both with and without keyword arguments.
    kernel.cz(control=qubit_1, target=qubit_0)
    kernel.ct(qubit_0, qubit_1)
    kernel.cs(qubit_1, qubit_0)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.x {{\[}}%[[VAL_2]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.y {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.z {{\[}}%[[VAL_2]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.t {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.s {{\[}}%[[VAL_2]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_kernel_ctrl_rotation():
    """
    Test the `cudaq.Kernel` on each controlled single qubit rotation gate/
    """
    kernel, angles = cudaq.make_kernel(list)
    assert kernel.argument_count == 1
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Check the overloads that accept a `QuakeValue`` as input.
    kernel.cr1(angles[0], qubit_0, qubit_1)
    kernel.crx(angles[1], qubit_1, qubit_0)
    kernel.cry(angles[2], qubit_0, qubit_1)
    kernel.crz(angles[3], qubit_1, qubit_0)

    # Check the overloads that take a `float` as input.
    kernel.cr1(0.0, qubit_1, qubit_0)
    kernel.crx(1.0, qubit_0, qubit_1)
    kernel.cry(2.0, qubit_1, qubit_0)
    kernel.crz(3.0, qubit_0, qubit_1)

    print(kernel)


# CHECK-LABEL: test_kernel_ctrl_rotation
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: (%[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3.000000e+00 : f64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_5]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_8:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
# CHECK:           quake.r1 (%[[VAL_10]]) {{\[}}%[[VAL_6]]] %[[VAL_7]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
# CHECK:           quake.rx (%[[VAL_12]]) {{\[}}%[[VAL_7]]] %[[VAL_6]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_13]] : !cc.ptr<f64>
# CHECK:           quake.ry (%[[VAL_14]]) {{\[}}%[[VAL_6]]] %[[VAL_7]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
# CHECK:           quake.rz (%[[VAL_16]]) {{\[}}%[[VAL_7]]] %[[VAL_6]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.r1 (%[[VAL_4]]) {{\[}}%[[VAL_7]]] %[[VAL_6]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rx (%[[VAL_3]]) {{\[}}%[[VAL_6]]] %[[VAL_7]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_2]]) {{\[}}%[[VAL_7]]] %[[VAL_6]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_1]]) {{\[}}%[[VAL_6]]] %[[VAL_7]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_kernel_multi_ctrl():
    """
    Test the `cudaq.Kernel` on each multi-qubit gate (multi-controlled single
    qubit gates). We do this for the case of a 3-qubit kernel. 
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 3.
    qreg = kernel.qalloc(3)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    qubit_2 = qreg[2]
    # Apply each gate to entire register.
    # Note: we alternate between orders to make the circuit less trivial.
    kernel.ch([qubit_0, qubit_1], qubit_2)
    kernel.cx([qubit_2, qubit_0], qubit_1)
    kernel.cy([qubit_1, qubit_2], qubit_0)
    kernel.cz([qubit_0, qubit_1], qubit_2)
    kernel.ct([qubit_2, qubit_0], qubit_1)
    kernel.cs([qubit_1, qubit_2], qubit_0)
    kernel()
    assert (kernel.arguments == [])
    assert (kernel.argument_count == 0)
    print(kernel)


# CHECK-LABEL: test_kernel_multi_ctrl
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.h {{\[}}%[[VAL_1]], %[[VAL_2]]] %[[VAL_3]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.x {{\[}}%[[VAL_3]], %[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.y {{\[}}%[[VAL_2]], %[[VAL_3]]] %[[VAL_1]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.z {{\[}}%[[VAL_1]], %[[VAL_2]]] %[[VAL_3]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.t {{\[}}%[[VAL_3]], %[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.s {{\[}}%[[VAL_2]], %[[VAL_3]]] %[[VAL_1]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_kernel_ctrl_register():
    """
    Test the :class:`Kernel` on each control gate, providing a qubit
    register as the controls for the operation.
    """
    kernel = cudaq.make_kernel()
    # Allocate two qubit registers.
    controls = kernel.qalloc(3)
    targets = kernel.qalloc(2)
    qubit_0 = targets[0]
    qubit_1 = targets[1]

    # Test the gates both with and without keyword arguments.
    kernel.ch(control=controls, target=qubit_0)
    kernel.cx(controls, target=qubit_1)
    kernel.cy(controls, qubit_0)
    kernel.cz(controls, qubit_1)
    kernel.ct(controls, qubit_0)
    kernel.cs(controls, qubit_1)

    print(kernel)


# CHECK-LABEL: test_kernel_ctrl_register
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK-DAG:       %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h {{\[}}%[[VAL_0]]] %[[VAL_2]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.x {{\[}}%[[VAL_0]]] %[[VAL_3]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.y {{\[}}%[[VAL_0]]] %[[VAL_2]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.z {{\[}}%[[VAL_0]]] %[[VAL_3]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.t {{\[}}%[[VAL_0]]] %[[VAL_2]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.s {{\[}}%[[VAL_0]]] %[[VAL_3]] : (!quake.veq<3>, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_kernel_rotation_ctrl_register():
    """
    Test the :class:`Kernel` on each controlled rotation gate, providing a
    qubit register as the controls for the operation.
    """
    kernel, angles = cudaq.make_kernel(list)
    # Allocate two qubit registers.
    controls = kernel.qalloc(3)
    targets = kernel.qalloc(2)
    qubit_0 = targets[0]
    qubit_1 = targets[1]

    # Place the control register in the 1-state.
    kernel.x(controls)

    # Test the gates both with and without keyword arguments.
    # Using the `float` parameter overload here.
    kernel.cr1(0.0, controls, qubit_0)
    kernel.crx(1.0, controls, qubit_1)
    kernel.cry(2.0, controls, qubit_0)
    kernel.crz(3.0, controls, qubit_1)

    # Using the `QuakeValue` parameter overload here.
    kernel.cr1(angles[0], controls, qubit_0)
    kernel.crx(angles[1], controls, qubit_1)
    kernel.cry(angles[2], controls, qubit_0)
    kernel.crz(angles[3], controls, qubit_1)

    print(kernel)


# CHECK-LABEL: test_kernel_rotation_ctrl_register
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:      (%[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f64
# CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_9:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_9]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_9]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_12:.*]] = cc.loop while ((%[[VAL_13:.*]] = %[[VAL_7]]) -> (i64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_13]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_15]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_16]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_15]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_6]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.r1 (%[[VAL_5]]) {{\[}}%[[VAL_8]]] %[[VAL_10]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.rx (%[[VAL_4]]) {{\[}}%[[VAL_8]]] %[[VAL_11]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_3]]) {{\[}}%[[VAL_8]]] %[[VAL_10]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_2]]) {{\[}}%[[VAL_8]]] %[[VAL_11]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           %[[VAL_19:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:           %[[VAL_20:.*]] = cc.cast %[[VAL_19]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_21:.*]] = cc.load %[[VAL_20]] : !cc.ptr<f64>
# CHECK:           quake.r1 (%[[VAL_21]]) [%[[VAL_8]]] %[[VAL_10]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_19]][1] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_22]] : !cc.ptr<f64>
# CHECK:           quake.rx (%[[VAL_23]]) [%[[VAL_8]]] %[[VAL_11]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_19]][2] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_25:.*]] = cc.load %[[VAL_24]] : !cc.ptr<f64>
# CHECK:           quake.ry (%[[VAL_25]]) {{\[}}%[[VAL_8]]] %[[VAL_10]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           %[[VAL_26:.*]] = cc.compute_ptr %[[VAL_19]][3] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_27:.*]] = cc.load %[[VAL_26]] : !cc.ptr<f64>
# CHECK:           quake.rz (%[[VAL_27]]) {{\[}}%[[VAL_8]]] %[[VAL_11]] : (f64, !quake.veq<3>, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_ctrl_swap():
    """
    Tests the compilation of the various overloads of `cswap` gates.
    """
    kernel = cudaq.make_kernel()
    controls_vector = [kernel.qalloc() for _ in range(3)]
    controls_register = kernel.qalloc(3)
    first = kernel.qalloc()
    second = kernel.qalloc()

    kernel.cswap(controls_vector, first, second)
    kernel.cswap(controls_register, first, second)
    kernel.cswap([controls_vector[0], controls_vector[1], controls_register],
                 first, second)

    print(kernel)


# CHECK-LABEL: test_ctrl_swap
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK-DAG:       %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<3>
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
# CHECK:           quake.swap {{\[}}%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]] %[[VAL_4]], %[[VAL_5]] : (!quake.ref, !quake.ref, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.swap {{\[}}%[[VAL_3]]] %[[VAL_4]], %[[VAL_5]] : (!quake.veq<3>, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.swap {{\[}}%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]] %[[VAL_4]], %[[VAL_5]] : (!quake.ref, !quake.ref, !quake.veq<3>, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import numpy as np
import cudaq


def test_builder_look_up():
    """A custom operation can be looked up by its name in builder mode"""

    base_name = 'foo'
    op_count = 3

    def register_custom_operations(matrix):
        prev = np.identity(2)
        for t in range(op_count):
            new = prev @ matrix
            cudaq.register_operation(f'{base_name}_{t}', new)
            prev = new

    register_custom_operations(
        np.array([[1, 0], [0, np.exp(np.pi * 1j * 1 / 3)]]))

    kernel = cudaq.make_kernel()

    qubit = kernel.qalloc(1)
    ancilla = kernel.qalloc(2)

    kernel.x(qubit)
    kernel.h(ancilla)

    for i in range(op_count):
        kernel.__getattr__(f'{base_name}_{i}')(ancilla, qubit)

    print(kernel)
    counts = cudaq.sample(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<1>
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_8]]] : (!quake.veq<1>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_8]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_12:.*]] = cc.loop while ((%[[VAL_13:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_13]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_15]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_16]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_15]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.custom_op @__nvqpp__mlirgen__foo_0_generator_1.rodata {{\[}}%[[VAL_4]]] %[[VAL_3]] : (!quake.veq<2>, !quake.veq<1>) -> ()
# CHECK:           quake.custom_op @__nvqpp__mlirgen__foo_1_generator_1.rodata {{\[}}%[[VAL_4]]] %[[VAL_3]] : (!quake.veq<2>, !quake.veq<1>) -> ()
# CHECK:           quake.custom_op @__nvqpp__mlirgen__foo_2_generator_1.rodata {{\[}}%[[VAL_4]]] %[[VAL_3]] : (!quake.veq<2>, !quake.veq<1>) -> ()
# CHECK:           return
# CHECK:         }
# CHECK-DAG:         cc.global constant @__nvqpp__mlirgen__foo_0_generator_1.rodata (dense<[{{.*}}]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
# CHECK-DAG:         cc.global constant @__nvqpp__mlirgen__foo_1_generator_1.rodata (dense<[{{.*}}]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
# CHECK-DAG:         cc.global constant @__nvqpp__mlirgen__foo_2_generator_1.rodata (dense<[{{.*}}]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>

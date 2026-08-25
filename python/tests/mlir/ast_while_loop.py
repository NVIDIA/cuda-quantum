# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import numpy as np

import cudaq


def test_while():

    @cudaq.kernel
    def trowe():
        q = cudaq.qvector(6)
        i = 5
        while i > 0:
            ry(np.pi, q[i])
            i -= 1

    print(trowe)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__trowe..
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:       %[[CONSTANT_3:.*]] = arith.constant 5 : i64
# CHECK-DAG:       %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi ne, %[[VAL_0]], %[[CONSTANT_3]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_1:.*]]: i64):
# CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_3]], %[[VAL_1]] : i64
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[SUBI_0]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[CONSTANT_2]]) %[[EXTRACT_REF_0]] : (f64, !quake.ref) -> ()
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_0]] : i64
# CHECK:             cc.continue %[[ADDI_0]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:             cc.continue %[[VAL_2]] : i64
# CHECK:           } {normalized}
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<6>
# CHECK:           return
# CHECK:         }

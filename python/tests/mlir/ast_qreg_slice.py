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


def test_slice():

    # slice upper bound should be exclusive

    @cudaq.kernel
    def slice_qvec():
        q = cudaq.qvector(4)
        slice = q[2:]

        x(slice[0])
        x(slice[1])

        slice = q[:2]

        y(slice[0])
        y(slice[1])

        slice2 = q[1:3]  # should give q[1], q[2]

        z(slice2[0])
        z(slice2[1])

        # bad, will get mlir error
        # z(slice2[2])

        l = [1, 2, 3, 4, 5]
        subl = l[2:4]  # should give l[2] = 3, l[3] = 4
        for i, el in enumerate(subl):
            ry(el, q[i % q.size()])

        # Can get last qubit
        rz(np.pi, q[-1])

    slice_qvec()
    print(slice_qvec)


if __name__ == '__main__':
    test_slice()

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__slice_qvec
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 5 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_8:.*]] = quake.subveq %[[VAL_7]], 2, 3 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_10]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_11:.*]] = quake.subveq %[[VAL_7]], 0, 1 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_11]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_12]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_11]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_13]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_14:.*]] = quake.subveq %[[VAL_7]], 1, 2 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           return
# CHECK:         }

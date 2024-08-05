# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq



def test_control_on_adjoint():

    @cudaq.kernel
    def my_func(q: cudaq.qubit, theta: float):
        ry(theta, q)
        rz(theta, q)

    @cudaq.kernel
    def adj_func(q: cudaq.qubit, theta: float):
        cudaq.adjoint(my_func, q, theta)

    @cudaq.kernel
    def kernel(theta: float):
        ancilla = cudaq.qubit()
        q = cudaq.qubit()

        h(ancilla)
        cudaq.control(my_func, ancilla, q, theta)
        cudaq.control(adj_func, ancilla, q, theta)

    print(kernel)
    theta = 1.5
    # also test that this compiles and runs
    cudaq.sample(kernel, theta).dump()


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__my_func(
# CHECK-SAME:                                         %[[VAL_0:.*]]: !quake.ref,
# CHECK-SAME:                                         %[[VAL_1:.*]]: f64) {
# CHECK:           %[[VAL_2:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
# CHECK:           quake.ry (%[[VAL_3]]) %[[VAL_0]] : (f64, !quake.ref) -> ()
# CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
# CHECK:           quake.rz (%[[VAL_4]]) %[[VAL_0]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__adj_func(
# CHECK-SAME:                                          %[[VAL_0:.*]]: !quake.ref,
# CHECK-SAME:                                          %[[VAL_1:.*]]: f64) {
# CHECK:           %[[VAL_2:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen__my_func %[[VAL_0]], %[[VAL_3]] : (!quake.ref, f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<f64>
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
# CHECK:           quake.apply @__nvqpp__mlirgen__my_func {{\[}}%[[VAL_2]]] %[[VAL_3]], %[[VAL_4]] : (!quake.ref, !quake.ref, f64) -> ()
# CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
# CHECK:           quake.apply @__nvqpp__mlirgen__adj_func {{\[}}%[[VAL_2]]] %[[VAL_3]], %[[VAL_5]] : (!quake.ref, !quake.ref, f64) -> ()
# CHECK:           return
# CHECK:         }

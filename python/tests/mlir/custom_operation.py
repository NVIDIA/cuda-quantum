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


def test_bell_pair():

    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    print(bell)


def test_custom_adjoint():

    cudaq.register_operation("custom_s", np.array([1, 0, 0, 1j]))

    cudaq.register_operation("custom_s_adj", np.array([1, 0, 0, -1j]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        custom_s.adj(q)
        custom_s_adj(q)
        h(q)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__bell
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.custom_op @__nvqpp__mlirgen__custom_h_generator_1.rodata %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.custom_op @__nvqpp__mlirgen__custom_x_generator_1.rodata {{\[}}%[[VAL_2]]] %[[VAL_3]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.veq<2>
# CHECK:           return
# CHECK:         }
# CHECK:         cc.global constant private @__nvqpp__mlirgen__custom_h_generator_1.rodata (dense<[(0.70710678118654746,0.000000e+00), (0.70710678118654746,0.000000e+00), (0.70710678118654746,0.000000e+00), (-0.70710678118654746,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
# CHECK:         cc.global constant private @__nvqpp__mlirgen__custom_x_generator_1.rodata (dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.custom_op @__nvqpp__mlirgen__custom_s_generator_1.rodata<adj> %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.custom_op @__nvqpp__mlirgen__custom_s_adj_generator_1.rodata %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref
# CHECK:           return
# CHECK:         }
# CHECK:         cc.global constant private @__nvqpp__mlirgen__custom_s_generator_1.rodata (dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
# CHECK:         cc.global constant private @__nvqpp__mlirgen__custom_s_adj_generator_1.rodata (dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-0.000000e+00,-1.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>

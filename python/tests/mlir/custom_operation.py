# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest
import numpy as np
import cudaq


def test_bell_pair():

    cudaq.register_operation("custom_h", 1, 0,
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", 1, 0, np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    print(bell)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__bell() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.custom_op @custom_h.generator %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.custom_op @custom_x.generator {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
# CHECK:         cc.global constant @custom_h.generator ([array<f64: 0.70710678118654746, 0.000000e+00>, array<f64: 0.70710678118654746, 0.000000e+00>, array<f64: 0.70710678118654746, 0.000000e+00>, array<f64: -0.70710678118654746, 0.000000e+00>]) : !cc.array<complex<f64> x 4>
# CHECK:         cc.global constant @custom_x.generator ([array<f64: 0.000000e+00, 0.000000e+00>, array<f64: 1.000000e+00, 0.000000e+00>, array<f64: 1.000000e+00, 0.000000e+00>, array<f64: 0.000000e+00, 0.000000e+00>]) : !cc.array<complex<f64> x 4>

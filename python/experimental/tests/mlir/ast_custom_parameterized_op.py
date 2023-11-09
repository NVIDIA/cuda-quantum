# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../../python_packages/cudaq pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_custom_parameterized_op():
    custom_ry = cudaq.register_operation(lambda param: np.array([[
        np.cos(param / 2), -np.sin(param / 2)
    ], [np.sin(param / 2), np.cos(param / 2)]]))

    @cudaq.kernel(jit=True)
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        custom_ry(theta, q[1])
        x.ctrl(q[1], q[0])
    print(ansatz)

# CHECK-LABEL:     func.func @__nvqpp__mlirgen__ansatz 
# CHECK:           %[[VAL_0:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_1:.*]], %[[VAL_0]] : !cc.ptr<f64>
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_0]] : !cc.ptr<f64>
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_6:.*]] = call @custom_ry(%[[VAL_4]]) : (f64) -> !cc.stdvec<complex<f64>>
# CHECK:           quake.unitary %[[VAL_5]](%[[VAL_6]]) : (!quake.ref, !cc.stdvec<complex<f64>>) -> () {opName = "custom_ry"}
# CHECK:           quake.x {{\[}}%[[VAL_5]]] %[[VAL_3]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return

# CHECK-LABEL:     func.func @custom_ry
# CHECK:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f64
# CHECK:           %[[VAL_1:.*]] = arith.constant 16 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant -1.000000e+00 : f64
# CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
# CHECK:           %[[VAL_5:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_6:.*]], %[[VAL_5]] : !cc.ptr<f64>
# CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
# CHECK:           %[[VAL_8:.*]] = arith.divf %[[VAL_7]], %[[VAL_0]] : f64
# CHECK:           %[[VAL_9:.*]] = complex.create %[[VAL_8]], %[[VAL_4]] : complex<f64>
# CHECK:           %[[VAL_10:.*]] = complex.cos %[[VAL_9]] : complex<f64>
# CHECK:           %[[VAL_11:.*]] = complex.sin %[[VAL_9]] : complex<f64>
# CHECK:           %[[VAL_12:.*]] = complex.create %[[VAL_3]], %[[VAL_4]] : complex<f64>
# CHECK:           %[[VAL_13:.*]] = complex.mul %[[VAL_12]], %[[VAL_11]] : complex<f64>
# CHECK:           %[[VAL_14:.*]] = cc.alloca !cc.array<complex<f64> x 4>
# CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_14]][0] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
# CHECK:           cc.store %[[VAL_10]], %[[VAL_15]] : !cc.ptr<complex<f64>>
# CHECK:           %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_14]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
# CHECK:           cc.store %[[VAL_13]], %[[VAL_16]] : !cc.ptr<complex<f64>>
# CHECK:           %[[VAL_17:.*]] = cc.compute_ptr %[[VAL_14]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
# CHECK:           cc.store %[[VAL_11]], %[[VAL_17]] : !cc.ptr<complex<f64>>
# CHECK:           %[[VAL_18:.*]] = cc.compute_ptr %[[VAL_14]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
# CHECK:           cc.store %[[VAL_10]], %[[VAL_18]] : !cc.ptr<complex<f64>>
# CHECK:           %[[VAL_19:.*]] = cc.cast %[[VAL_14]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_20:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_19]], %[[VAL_2]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
# CHECK:           %[[VAL_21:.*]] = cc.stdvec_init %[[VAL_20]], %[[VAL_2]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<complex<f64>>
# CHECK:           return %[[VAL_21]] : !cc.stdvec<complex<f64>>
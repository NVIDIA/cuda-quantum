/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt --canonicalize --cc-loop-normalize --cc-loop-unroll | FileCheck %s

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    const std::vector<cudaq::complex> stateVector{1.0, 0.0, 0.0, 0.0};
    cudaq::qvector v(stateVector);
    h(v);
    mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel() attributes
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = complex.create %[[VAL_0]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_3:.*]] = complex.create %[[VAL_1]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_4:.*]] = complex.create %[[VAL_1]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_5:.*]] = complex.create %[[VAL_1]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][0] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_6]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_8]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_6]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_6]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_10]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_13:.*]] = quake.init_state %[[VAL_12]], %[[VAL_11]] : (!quake.veq<2>, !cc.ptr<complex<f64>>) -> !quake.veq<2>
// CHECK:           %[[VAL_14:.*]] = quake.extract_ref %[[VAL_13]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_14]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_13]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_15]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_16:.*]] = quake.mz %[[VAL_13]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

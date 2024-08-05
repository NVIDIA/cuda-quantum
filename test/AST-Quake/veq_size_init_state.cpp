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
// CHECK:           %[[VAL_0:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK:           %[[VAL_1:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_2]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_2]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_6]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_9:.*]] = quake.init_state %[[VAL_8]], %[[VAL_7]] : (!quake.veq<2>, !cc.ptr<complex<f64>>) -> !quake.veq<2>
// CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_9]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_10]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_9]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_11]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_9]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cc-loop-normalize --cc-loop-unroll | FileCheck %s

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    const std::vector<cudaq::complex> stateVector{1.0, 0.0, 0.0, 0.0};
    cudaq::qvector v(stateVector);
    h(v);
    mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_8]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_5]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_10]], %[[VAL_4]] : (!cc.ptr<complex<f64>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_16:.*]] = cc.loop while ((%[[VAL_17:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_17]], %[[VAL_15]] : i64
// CHECK:             cc.condition %[[VAL_18]](%[[VAL_17]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
// CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_14]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_20]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_19]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i64):
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_22]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_23:.*]] = quake.mz %[[VAL_14]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }


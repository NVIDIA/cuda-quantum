/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct Colonel {
  void operator()(const std::vector<double> &parms) __qpu__ {
    cudaq::qubit q;
    double sum = 0.0;

    for (auto &d : parms) {
      sum += d;
    }

    // The above loop is syntactic sugar for the following loop.
    //
    // for (std::size_t i = 0; i < parms.size(); i++) {
    //   const double d = parms[i]; // read-only copy
    //   sum += d;
    // }

    rx(sum, q);
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Colonel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:        %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : index
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_16:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = arith.addf %[[VAL_16]], %[[VAL_15]] : f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_12]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: index):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_19]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_20]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> i1

struct Lt_Colonel {
  void operator()(const std::vector<double> &parms) __qpu__ {
    cudaq::qubit q;
    double sum = 0.0;

    for (auto &d : parms) {
      sum = sum + d;
    }

    // The above loop is syntactic sugar for the following loop.
    //
    // for (std::size_t i = 0; i < parms.size(); i++) {
    //   const double &d = parms[i];
    //   sum += d;
    // }

    rx(sum, q);
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Lt_Colonel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : index
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_15:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_16:.*]] = cc.load %[[VAL_14]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = arith.addf %[[VAL_15]], %[[VAL_16]] : f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_12]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: index):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_19]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_20]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> i1

struct Qernel {
  void operator()(std::vector<double> parms) __qpu__ {
    cudaq::qubit q;
    double sum = 0.0;

    for (auto &d : parms) {
      d = sqrt(d);
    }

    // The above loop is syntactic sugar for the following loop.
    //
    // for (std::size_t i = 0; i < parms.size(); i++) {
    //   double &d = parms[i];
    //   d = sqrt(d);  // assignment to values held by parms
    // }

    for (auto d : parms) {
      sum = sum + d;
      d += 1000.0;
    }

    // The above loop is syntactic sugar for the following loop.
    //
    // for (std::size_t i = 0; i < parms.size(); i++) {
    //   double d = parms[i]; // copy
    //   sum += d;
    //   d = d + 1000.0; // dead value; parms not changed
    // }

    rx(sum, q);
    mz(q);
  }
};
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1.000000e+03 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_6:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_7]] : i64 to index
// CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_9]] : index
// CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: index):
// CHECK:             %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : index to i64
// CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_8]]{{\[}}%[[VAL_14]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = func.call @sqrt(%[[VAL_16]]) : (f64) -> f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_15]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_13]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: index):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_19]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_21:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_22:.*]] = arith.index_cast %[[VAL_20]] : i64 to index
// CHECK:           %[[VAL_23:.*]] = cc.loop while ((%[[VAL_24:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_22]] : index
// CHECK:             cc.condition %[[VAL_25]](%[[VAL_24]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_26:.*]]: index):
// CHECK:             %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : index to i64
// CHECK:             cc.scope {
// CHECK:               %[[VAL_28:.*]] = cc.compute_ptr %[[VAL_21]]{{\[}}%[[VAL_27]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:               %[[VAL_29:.*]] = cc.alloca f64
// CHECK:               %[[VAL_30:.*]] = cc.load %[[VAL_28]] : !cc.ptr<f64>
// CHECK:               cc.store %[[VAL_30]], %[[VAL_29]] : !cc.ptr<f64>
// CHECK-DAG:           %[[VAL_31:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK-DAG:           %[[VAL_32:.*]] = cc.load %[[VAL_29]] : !cc.ptr<f64>
// CHECK:               %[[VAL_33:.*]] = arith.addf %[[VAL_31]], %[[VAL_32]] : f64
// CHECK:               cc.store %[[VAL_33]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_29]] : !cc.ptr<f64>
// CHECK:               %[[VAL_35:.*]] = arith.addf %[[VAL_34]], %[[VAL_1]] : f64
// CHECK:               cc.store %[[VAL_35]], %[[VAL_29]] : !cc.ptr<f64>
// CHECK:             }
// CHECK:             cc.continue %[[VAL_26]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_36:.*]]: index):
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_37]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_38:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_38]]) %[[VAL_5]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_39:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> i1

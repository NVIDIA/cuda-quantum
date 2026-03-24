/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Colonel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:        %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i64
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
// CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_12]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_16:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = arith.addf %[[VAL_16]], %[[VAL_15]] : f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_12]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_19]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_20]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> !quake.measure
// clang-format on

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Lt_Colonel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i64
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
// CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_12]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_15:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK-DAG:         %[[VAL_16:.*]] = cc.load %[[VAL_14]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = arith.addf %[[VAL_15]], %[[VAL_16]] : f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_12]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_19]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_20]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> !quake.measure
// clang-format on

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1.000000e+03 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_6:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_7]] : i64
// CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_8]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
// CHECK:             %[[VAL_17:.*]] = math.sqrt %[[VAL_16]] : f64
// CHECK:             cc.store %[[VAL_17]], %[[VAL_15]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_13]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_19]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_20:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_21:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_23:.*]] = cc.loop while ((%[[VAL_24:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_20]] : i64
// CHECK:             cc.condition %[[VAL_25]](%[[VAL_24]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_26:.*]]: i64):
// CHECK:             cc.scope {
// CHECK:               %[[VAL_28:.*]] = cc.compute_ptr %[[VAL_21]]{{\[}}%[[VAL_26]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
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
// CHECK:             cc.continue %[[VAL_26]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_36:.*]]: i64):
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_37]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_38:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_38]]) %[[VAL_5]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_39:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> !quake.measure
// clang-format on

struct Nesting {
  void operator()(std::vector<double> parms,
                  const std::vector<float> &otherParms) __qpu__ {
    cudaq::qubit q;
    double sum = 0.0;

    for (auto &d : parms) {
      for (auto &f : otherParms) {
        sum += d * f;
      }
    }

    rx(sum, q);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Nesting(
// CHECK-SAME:        %[[VAL_0:.*]]: !cc.stdvec<f64>, %[[VAL_1:.*]]: !cc.stdvec<f32>) attributes
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_6:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_7]] : i64
// CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_8]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_16:.*]] = cc.stdvec_size %[[VAL_1]] : (!cc.stdvec<f32>) -> i64
// CHECK:             %[[VAL_17:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<f32>) -> !cc.ptr<!cc.array<f32 x ?>>
// CHECK:             %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:               %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_16]] : i64
// CHECK:               cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_22:.*]]: i64):
// CHECK:               %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_17]]{{\[}}%[[VAL_22]]] : (!cc.ptr<!cc.array<f32 x ?>>, i64) -> !cc.ptr<f32>
// CHECK-DAG:           %[[VAL_25:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
// CHECK-DAG:           %[[VAL_26:.*]] = cc.load %[[VAL_24]] : !cc.ptr<f32>
// CHECK:               %[[VAL_27:.*]] = cc.cast %[[VAL_26]] : (f32) -> f64
// CHECK:               %[[VAL_28:.*]] = arith.mulf %[[VAL_25]], %[[VAL_27]] : f64
// CHECK:               %[[VAL_29:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:               %[[VAL_30:.*]] = arith.addf %[[VAL_29]], %[[VAL_28]] : f64
// CHECK:               cc.store %[[VAL_30]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:               cc.continue %[[VAL_22]] : i64
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_31:.*]]: i64):
// CHECK:               %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_2]] : i64
// CHECK:               cc.continue %[[VAL_32]] : i64
// CHECK:             } {invariant}
// CHECK:             cc.continue %[[VAL_13]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i64):
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_34]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_35:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_35]]) %[[VAL_5]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_36:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct FreeRange {
  void operator()(cudaq::qvector<> r, unsigned N) __qpu__ {
    for (auto i : cudaq::range(static_cast<int>(N))) {
      h(r[i]);
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__FreeRange(
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: i32) attributes
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.cast unsigned %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_6]] : i64
// CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_11]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_15]] : i64
// CHECK:           } {invariant}
// clang-format on

struct FreeRangeChicken {
  void operator()(cudaq::qvector<> r, int Z, int N, int S) __qpu__ {
    for (auto i : cudaq::range(Z, N, S)) {
      h(r[i]);
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__FreeRangeChicken(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>,
// CHECK-SAME:      %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32)
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.cast signed %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[VAL_13:.*]] = cc.cast signed %[[VAL_10]] : (i32) -> i64
// CHECK:           %[[VAL_14:.*]] = cc.cast signed %[[VAL_11]] : (i32) -> i64
// CHECK:           %[[VAL_16:.*]] = call @__nvqpp_CudaqSizeFromTriple(%[[VAL_12]], %[[VAL_13]], %[[VAL_14]]) : (i64, i64, i64) -> i64
// CHECK:           %[[VAL_17:.*]]:2 = cc.loop while ((%[[VAL_18:.*]] = %[[VAL_5]], %[[VAL_19:.*]] = %[[VAL_12]]) -> (i64, i64)) {
// CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_16]] : i64
// CHECK:             cc.condition %[[VAL_20]](%[[VAL_18]], %[[VAL_19]] : i64, i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: i64):
// CHECK:             %[[VAL_23:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_21]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_23]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_21]], %[[VAL_22]] : i64, i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_24:.*]]: i64, %[[VAL_25:.*]]: i64):
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_24]], %[[VAL_4]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_14]] : i64
// CHECK:             cc.continue %[[VAL_26]], %[[VAL_27]] : i64, i64
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }
// clang-format on

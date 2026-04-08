/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ std::uint64_t
qubit_values_to_integer(const std::vector<cudaq::measure_result> &values) {
  std::uint64_t result = 0;

  for (int64_t i = 0; i < values.size(); i++) {
    result |= static_cast<std::uint64_t>(values[i]) << i;
  }

  return result;
}

__qpu__ uint64_t foo() {
  cudaq::qvector v(2);
  x(v);
  auto results = mz(v);
  return qubit_values_to_integer(results);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qubit_values_to_integer.
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.measurements<?>) -> i64 attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_6:.*]] = quake.measurements_size %[[VAL_0]] : (!quake.measurements<?>) -> i64
// CHECK:               %[[VAL_7:.*]] = arith.cmpi ult, %[[VAL_5]], %[[VAL_6]] : i64
// CHECK:               cc.condition %[[VAL_7]]
// CHECK:             } do {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_9:.*]] = quake.get_measure %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.measurements<?>, i64) -> !quake.measure
// CHECK:               %[[VAL_10:.*]] = quake.discriminate %[[VAL_9]] : (!quake.measure) -> i1
// CHECK:               %[[VAL_11:.*]] = cc.cast unsigned %[[VAL_10]] : (i1) -> i64
// CHECK:               %[[VAL_12:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_13:.*]] = arith.shli %[[VAL_11]], %[[VAL_12]] : i64
// CHECK:               %[[VAL_14:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_15:.*]] = arith.ori %[[VAL_14]], %[[VAL_13]] : i64
// CHECK:               cc.store %[[VAL_15]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i64
// CHECK:               cc.store %[[VAL_17]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:           return %[[VAL_18]] : i64
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo.
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<2>, i64) -> !quake.ref
// CHECK:             quake.x %[[VAL_8]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_10]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_11:.*]] = quake.mz %[[VAL_3]] name "results" : (!quake.veq<2>) -> !quake.measurements<2>
// CHECK:           %[[VAL_12:.*]] = quake.relax_size %[[VAL_11]] : (!quake.measurements<2>) -> !quake.measurements<?>
// CHECK:           %[[VAL_13:.*]] = call @__nvqpp__mlirgen__function_qubit_values_to_integer.
// CHECK:           return %[[VAL_13]] : i64
// CHECK:         }
// clang-format on

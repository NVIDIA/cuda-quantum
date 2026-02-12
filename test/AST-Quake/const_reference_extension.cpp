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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qubit_values_to_integer._Z23qubit_values_to_integerRKSt6vectorIN5cudaq14measure_resultESaIS1_EE(
// CHECK-SAME:                                                                                                                                          %[[VAL_0:.*]]: !cc.stdvec<!quake.measure>) -> i64 attributes {"cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:               %[[VAL_7:.*]] = arith.cmpi ult, %[[VAL_5]], %[[VAL_6]] : i64
// CHECK:               cc.condition %[[VAL_7]]
// CHECK:             } do {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_9:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>>
// CHECK:               %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_8]]] : (!cc.ptr<!cc.array<!quake.measure x ?>>, i64) -> !cc.ptr<!quake.measure>
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<!quake.measure>
// CHECK:               %[[VAL_12:.*]] = quake.discriminate %[[VAL_11]] : (!quake.measure) -> i1
// CHECK:               %[[VAL_13:.*]] = cc.cast unsigned %[[VAL_12]] : (i1) -> i64
// CHECK:               %[[VAL_14:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_15:.*]] = arith.shli %[[VAL_13]], %[[VAL_14]] : i64
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_17:.*]] = arith.ori %[[VAL_16]], %[[VAL_15]] : i64
// CHECK:               cc.store %[[VAL_17]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : i64
// CHECK:               cc.store %[[VAL_19]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:           return %[[VAL_20]] : i64
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo._Z3foov() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
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
// CHECK:           %[[VAL_11:.*]] = quake.mz %[[VAL_3]] name "results" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_12:.*]] = call @__nvqpp__mlirgen__function_qubit_values_to_integer.{{.*}}(%[[VAL_11]]) : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           return %[[VAL_12]] : i64
// CHECK:         }

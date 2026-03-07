/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ int test_kernel(int count) {
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
  auto results = mz(v);
  int acc = 0;
  for (auto result : results)
    acc += (result ? 1 : 0);
  return acc;
}

int main() {
  constexpr int numQubits = 4;
  auto results = cudaq::run(100, test_kernel, numQubits);
  if (results.size() != 100) {
    printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
  } else {
    for (auto result : results)
      std::cout << "Result: " << result << "\n";
  }
  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_kernel._Z11test_kerneli(
// CHECK-SAME:                                                                       %[[VAL_0:.*]]: i32) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_7]] : i64]
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.h %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_10:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_4]], %[[VAL_10]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:               %[[VAL_12:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_13]] : i32
// CHECK:               cc.condition %[[VAL_14]]
// CHECK:             } do {
// CHECK:               %[[VAL_15:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:               %[[VAL_16:.*]] = cc.cast signed %[[VAL_15]] : (i32) -> i64
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_20:.*]] = cc.cast signed %[[VAL_19]] : (i32) -> i64
// CHECK:               %[[VAL_21:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_17]]] %[[VAL_21]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_22:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : i32
// CHECK:               cc.store %[[VAL_23]], %[[VAL_10]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = quake.mz %[[VAL_8]] name "results" : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_25:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = cc.stdvec_size %[[VAL_24]] : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           %[[VAL_27:.*]] = cc.stdvec_data %[[VAL_24]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>>
// CHECK:           %[[VAL_28:.*]] = cc.loop while ((%[[VAL_29:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_29]], %[[VAL_26]] : i64
// CHECK:             cc.condition %[[VAL_30]](%[[VAL_29]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_31:.*]]: i64):
// CHECK:             cc.scope {
// CHECK:               %[[VAL_32:.*]] = cc.compute_ptr %[[VAL_27]]{{\[}}%[[VAL_31]]] : (!cc.ptr<!cc.array<!quake.measure x ?>>, i64) -> !cc.ptr<!quake.measure>
// CHECK:               %[[VAL_33:.*]] = cc.alloca !quake.measure
// CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_32]] : !cc.ptr<!quake.measure>
// CHECK:               cc.store %[[VAL_34]], %[[VAL_33]] : !cc.ptr<!quake.measure>
// CHECK:               %[[VAL_35:.*]] = cc.load %[[VAL_33]] : !cc.ptr<!quake.measure>
// CHECK:               %[[VAL_36:.*]] = quake.discriminate %[[VAL_35]] : (!quake.measure) -> i1
// CHECK:               %[[VAL_37:.*]] = cc.if(%[[VAL_36]]) -> i32 {
// CHECK:                 cc.continue %[[VAL_3]] : i32
// CHECK:               } else {
// CHECK:                 cc.continue %[[VAL_4]] : i32
// CHECK:               }
// CHECK:             }
// CHECK:             cc.continue %{{.*}} : i64

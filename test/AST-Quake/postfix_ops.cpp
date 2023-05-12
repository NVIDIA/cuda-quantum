/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

void uma(cudaq::qubit&,cudaq::qubit&,cudaq::qubit&);

__qpu__ void test(cudaq::qspan<> a, cudaq::qspan<> b) {
  for (uint32_t i = a.size(); i-- > 1ul; 0) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test.
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.qvec<?>,
// CHECK-SAME:         %[[VAL_1:.*]]: !quake.qvec<?>) attributes {"cudaq-kernel"} {
// CHECK:           cc.scope {
// CHECK:             %[[VAL_2:.*]] = quake.vec_size %[[VAL_0]] : (!quake.qvec<?>) -> i64
// CHECK:             %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i64 to i32
// CHECK:             %[[VAL_4:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_3]], %[[VAL_4]][] : memref<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_5:.*]] = memref.load %[[VAL_4]][] : memref<i32>
// CHECK:               %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:               memref.store %[[VAL_7]], %[[VAL_4]][] : memref<i32>
// CHECK:               %[[VAL_8:.*]] = arith.extui %[[VAL_5]] : i32 to i64
// CHECK:               %[[VAL_9:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_10:.*]] = arith.cmpi ugt, %[[VAL_8]], %[[VAL_9]] : i64
// CHECK:               cc.condition %[[VAL_10]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_11:.*]] = memref.load %[[VAL_4]][] : memref<i32>
// CHECK:                 %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:                 %[[VAL_13:.*]] = arith.constant 1 : i64
// CHECK:                 %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:                 %[[VAL_15:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_14]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_16:.*]] = memref.load %[[VAL_4]][] : memref<i32>
// CHECK:                 %[[VAL_17:.*]] = arith.extui %[[VAL_16]] : i32 to i64
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : i64
// CHECK:                 %[[VAL_19:.*]] = arith.subi %[[VAL_17]], %[[VAL_18]] : i64
// CHECK:                 %[[VAL_20:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_19]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_21:.*]] = memref.load %[[VAL_4]][] : memref<i32>
// CHECK:                 %[[VAL_22:.*]] = arith.extui %[[VAL_21]] : i32 to i64
// CHECK:                 %[[VAL_23:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_22]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_24:.*]] = arith.constant 0 : i32
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

void uma(cudaq::qubit&,cudaq::qubit&,cudaq::qubit&);

__qpu__ void test1(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (1) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  }
}

__qpu__ void test2(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (true) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  }
}

__qpu__ void test3(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  do {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  } while (false);
}

__qpu__ double test4(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  double trouble = i;
  do {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  } while (0);
  return trouble;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test1.
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>)
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             cc.condition %[[VAL_2]]
// CHECK:           } do {
// CHECK:             %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_8:.*]] = arith.extui %[[VAL_7]] : i32 to i64
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_3]] : i64
// CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_11:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_3]] : i64
// CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             func.call @_Z3umaRN5cudaq5quditILm2EEES2_S2_(%[[VAL_10]], %[[VAL_14]], %[[VAL_17]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test2.
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>) attributes
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             cc.condition %[[VAL_3]]
// CHECK:           } do {
// CHECK:             %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_8:.*]] = arith.extui %[[VAL_7]] : i32 to i64
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_11:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             func.call @_Z3umaRN5cudaq5quditILm2EEES2_S2_(%[[VAL_10]], %[[VAL_14]], %[[VAL_17]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test3.
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>) attributes
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant false
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.loop do {
// CHECK:             %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_8:.*]] = arith.extui %[[VAL_7]] : i32 to i64
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_11:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             func.call @_Z3umaRN5cudaq5quditILm2EEES2_S2_(%[[VAL_10]], %[[VAL_14]], %[[VAL_17]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:             cc.continue
// CHECK:           } while {
// CHECK:             cc.condition %[[VAL_3]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test4.
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>) -> f64 attributes
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = arith.uitofp %[[VAL_7]] : i32 to f64
// CHECK:           %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           cc.loop do {
// CHECK:             %[[VAL_10:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_11:.*]] = arith.extui %[[VAL_10]] : i32 to i64
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_3]] : i64
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_14:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_15:.*]] = arith.extui %[[VAL_14]] : i32 to i64
// CHECK:             %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_3]] : i64
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_18:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_19:.*]] = arith.extui %[[VAL_18]] : i32 to i64
// CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             func.call @_Z3umaRN5cudaq5quditILm2EEES2_S2_(%[[VAL_13]], %[[VAL_17]], %[[VAL_20]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:             cc.continue
// CHECK:           } while {
// CHECK:             cc.condition %[[VAL_2]]
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           return %[[VAL_21]] : f64
// CHECK:         }

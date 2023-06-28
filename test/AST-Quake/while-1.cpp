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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test1
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}})
// CHECK:           %[[VAL_2:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i64 to i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:             cc.condition %[[VAL_7]]
// CHECK:           } do {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.extui %[[VAL_8]] : i32 to i64
// CHECK:               %[[VAL_10:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i64
// CHECK:               %[[VAL_12:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_13:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_14:.*]] = arith.extui %[[VAL_13]] : i32 to i64
// CHECK:               %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_15]] : i64
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_19:.*]] = arith.extui %[[VAL_18]] : i32 to i64
// CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             }
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test2
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}})
// CHECK:           %[[VAL_2:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i64 to i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             %[[VAL_5:.*]] = arith.constant true
// CHECK:             cc.condition %[[VAL_5]]
// CHECK:           } do {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_7:.*]] = arith.extui %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
// CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:               %[[VAL_13:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:               %[[VAL_15:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_17:.*]] = arith.extui %[[VAL_16]] : i32 to i64
// CHECK:               %[[VAL_18:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_17]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             }
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test3
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}})
// CHECK:           %[[VAL_2:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i64 to i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.loop do {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_6:.*]] = arith.extui %[[VAL_5]] : i32 to i64
// CHECK:               %[[VAL_7:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_7]] : i64
// CHECK:               %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = arith.extui %[[VAL_10]] : i32 to i64
// CHECK:               %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
// CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_15:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             }
// CHECK:             cc.continue
// CHECK:           } while {
// CHECK:             %[[VAL_18:.*]] = arith.constant false
// CHECK:             cc.condition %[[VAL_18]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test4
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.veq<?>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) -> f64
// CHECK:           %[[VAL_2:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i64 to i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = arith.uitofp %[[VAL_5]] : i32 to f64
// CHECK:           %[[VAL_7:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           cc.loop do {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.extui %[[VAL_8]] : i32 to i64
// CHECK:               %[[VAL_10:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i64
// CHECK:               %[[VAL_12:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_13:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_14:.*]] = arith.extui %[[VAL_13]] : i32 to i64
// CHECK:               %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_15]] : i64
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_19:.*]] = arith.extui %[[VAL_18]] : i32 to i64
// CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             }
// CHECK:             cc.continue
// CHECK:           } while {
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_22]] : i32
// CHECK:             cc.condition %[[VAL_23]]
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           cc.return %[[VAL_24]] : f64
// CHECK:         }


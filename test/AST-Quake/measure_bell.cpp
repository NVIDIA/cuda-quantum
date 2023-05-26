/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct bell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qreg q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      bool r0 = results[0];
      if (r0 == results[1]) {
        n++;
      }
    }
  }
};

int main() { bell{}(100); }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bell(
// CHECK-SAME:      %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_6]][] : memref<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_8:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_7]], %[[VAL_8]][] : memref<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_10:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:               %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               cc.condition %[[VAL_11]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : i32 to i64
// CHECK:                 %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.h %[[VAL_14]] :
// CHECK:                 %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK:                 %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_19:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:                 %[[VAL_20:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.x [%[[VAL_17]]] %[[VAL_20]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:                 %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_23:.*]] = arith.extsi %[[VAL_22]] : i32 to i64
// CHECK:                 %[[VAL_24:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_24]]{{\[}}%[[VAL_23]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_27:.*]] = memref.alloca() : memref<i1>
// CHECK:                 memref.store %[[VAL_26]], %[[VAL_27]][] : memref<i1>
// CHECK:                 %[[VAL_28:.*]] = memref.load %[[VAL_27]][] : memref<i1>
// CHECK:                 %[[VAL_29:.*]] = arith.extui %[[VAL_28]] : i1 to i32
// CHECK:                 %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_31:.*]] = arith.extsi %[[VAL_30]] : i32 to i64
// CHECK:                 %[[VAL_32:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_33:.*]] = cc.compute_ptr %[[VAL_32]]{{\[}}%[[VAL_31]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_34:.*]] = cc.load %[[VAL_33]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_35:.*]] = arith.extui %[[VAL_34]] : i1 to i32
// CHECK:                 %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_35]] : i32
// CHECK:                 cc.if(%[[VAL_36]]) {
// CHECK:                   cc.scope {
// CHECK:                     %[[VAL_37:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK:                     %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:                     %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:                     memref.store %[[VAL_39]], %[[VAL_6]][] : memref<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_40:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:               memref.store %[[VAL_42]], %[[VAL_8]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

struct libertybell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qreg q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      if (results[0] == results[1]) {
        n++;
      }
    }
  }
};

struct tinkerbell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qreg q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      auto r0 = results[0];
      auto r1 = results[1];
      if (r0 == r1) {
        n++;
      }
    }
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__libertybell(
// CHECK-SAME:        %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_6]][] : memref<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_8:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_7]], %[[VAL_8]][] : memref<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_10:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:               %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               cc.condition %[[VAL_11]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : i32 to i64
// CHECK:                 %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.h %[[VAL_14]] :
// CHECK:                 %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK:                 %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_19:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:                 %[[VAL_20:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.x [%[[VAL_17]]] %[[VAL_20]] : (
// CHECK:                 %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:                 %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_23:.*]] = arith.extsi %[[VAL_22]] : i32 to i64
// CHECK:                 %[[VAL_24:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_24]][%[[VAL_23]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_28:.*]] = arith.extsi %[[VAL_27]] : i32 to i64
// CHECK:                 %[[VAL_29:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_30:.*]] = cc.compute_ptr %[[VAL_29]][%[[VAL_28]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_31:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_31]], %[[VAL_26]] : i1
// CHECK:                 cc.if(%[[VAL_32]]) {
// CHECK:                   cc.scope {
// CHECK:                     %[[VAL_33:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK:                     %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:                     %[[VAL_35:.*]] = arith.addi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:                     memref.store %[[VAL_35]], %[[VAL_6]][] : memref<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_36:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_36]], %[[VAL_37]] : i32
// CHECK:               memref.store %[[VAL_38]], %[[VAL_8]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__tinkerbell(
// CHECK-SAME:        %[[VAL_0:.*]]: i32) attributes
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_6]][] : memref<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_8:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_7]], %[[VAL_8]][] : memref<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_10:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:               %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               cc.condition %[[VAL_11]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : i32 to i64
// CHECK:                 %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.h %[[VAL_14]]
// CHECK:                 %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK:                 %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_19:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:                 %[[VAL_20:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[VAL_17]]] %[[VAL_20]] :
// CHECK:                 %[[VAL_21:.*]] = quake.mz %[[VAL_4]] : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:                 %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_23:.*]] = arith.extsi %[[VAL_22]] : i32 to i64
// CHECK:                 %[[VAL_24:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_24]][%[[VAL_23]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_27:.*]] = memref.alloca() : memref<i1>
// CHECK:                 memref.store %[[VAL_26]], %[[VAL_27]][] : memref<i1>
// CHECK:                 %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_29:.*]] = arith.extsi %[[VAL_28]] : i32 to i64
// CHECK:                 %[[VAL_30:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_31:.*]] = cc.compute_ptr %[[VAL_30]][%[[VAL_29]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:                 %[[VAL_32:.*]] = cc.load %[[VAL_31]] : !cc.ptr<i1>
// CHECK:                 %[[VAL_33:.*]] = memref.alloca() : memref<i1>
// CHECK:                 memref.store %[[VAL_32]], %[[VAL_33]][] : memref<i1>
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_33]][] : memref<i1>
// CHECK:                 %[[VAL_35:.*]] = memref.load %[[VAL_27]][] : memref<i1>
// CHECK:                 %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_35]] : i1
// CHECK:                 cc.if(%[[VAL_36]]) {
// CHECK:                   cc.scope {
// CHECK:                     %[[VAL_37:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK:                     %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:                     %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:                     memref.store %[[VAL_39]], %[[VAL_6]][] : memref<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_40:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:               %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:               memref.store %[[VAL_42]], %[[VAL_8]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


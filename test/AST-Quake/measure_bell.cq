/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --cse | FileCheck %s

#include <cudaq.h>

// These "bell" tests are very similar. Each tests a slightly different syntax
// for the equality test of the bits being measured.

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bell(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_6:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.h %[[VAL_10]] : (!quake.ref) -> ()
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_10]]] %[[VAL_11]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               %[[VAL_12:.*]] = quake.mz %[[VAL_4]] name "results" : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:               %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_12]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:               %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]][0] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK:               %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i1>
// CHECK:               %[[VAL_16:.*]] = cc.alloca i1
// CHECK:               cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<i1>
// CHECK:               %[[VAL_17:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i1>
// CHECK:               %[[VAL_18:.*]] = cc.compute_ptr %[[VAL_13]][1] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK:               %[[VAL_19:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i1>
// CHECK:               %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_19]] : i1
// CHECK:               cc.if(%[[VAL_20]]) {
// CHECK:                 %[[VAL_21:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_1]] : i32
// CHECK:                 cc.store %[[VAL_22]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_23:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_24]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__libertybell(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_6:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.h %[[VAL_10]] : (!quake.ref) -> ()
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_10]]] %[[VAL_11]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               %[[VAL_12:.*]] = quake.mz %[[VAL_4]] name "results" : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:               %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_12]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:               %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]][0] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_13]][1] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_17:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i1>
// CHECK:               %[[VAL_18:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_16]] : i1
// CHECK:               cc.if(%[[VAL_18]]) {
// CHECK:                 %[[VAL_19:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_1]] : i32
// CHECK:                 cc.store %[[VAL_20]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_21:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_22]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__tinkerbell(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_6:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.h %[[VAL_10]] : (!quake.ref) -> ()
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_10]]] %[[VAL_11]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               %[[VAL_12:.*]] = quake.mz %[[VAL_4]] name "results" : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:               %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_12]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:               %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]][0] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_13]][1] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i1>
// CHECK-DAG:           %[[VAL_17:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i1>
// CHECK:               %[[VAL_18:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_16]] : i1
// CHECK:               cc.if(%[[VAL_18]]) {
// CHECK:                 %[[VAL_19:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_1]] : i32
// CHECK:                 cc.store %[[VAL_20]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_21:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_22]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

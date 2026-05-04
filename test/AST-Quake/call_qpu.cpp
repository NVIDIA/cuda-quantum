/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt -memtoreg=quantum=0 -canonicalize | FileCheck %s

#include <cudaq.h>

std::vector<bool> func_achat(cudaq::qview<> &qv) __qpu__ {
  // measure the entire register
  return mz(qv);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_func_achat._Z10func_achatRN5cudaq5qviewILm2EEE(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) -> !cc.stdvec<i1> attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_6:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_4]], %[[VAL_5]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_init %[[VAL_6]], %[[VAL_5]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_7]] : !cc.stdvec<i1>
// CHECK:         }

int func_shiim(cudaq::qvector<> &qv) __qpu__ {
  auto vs = qv.slice(1, 3);
  auto bs = func_achat(vs);
  int i;
  for (auto b : bs) {
    if (b)
      ++i;
  }
  return i;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_func_shiim._Z10func_shiimRN5cudaq7qvectorILm2EEE(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) -> i32 attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = quake.subveq %[[VAL_0]], 1, 3 : (!quake.veq<?>) -> !quake.veq<3>
// CHECK:           %[[VAL_5:.*]] = quake.relax_size %[[VAL_4]] : (!quake.veq<3>) -> !quake.veq<?>
// CHECK:           %[[VAL_6:.*]] = call @__nvqpp__mlirgen__function_func_achat._Z10func_achatRN5cudaq5qviewILm2EEE(%[[VAL_5]]) : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_6]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_size %[[VAL_6]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_9:.*]] = cc.alloca i8{{\[}}%[[VAL_8]] : i64]
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_10]], %[[VAL_7]], %[[VAL_8]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.undef i32
// CHECK:           %[[VAL_13:.*]]:2 = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_2]], %[[VAL_15:.*]] = %[[VAL_11]]) -> (i64, i32)) {
// CHECK:             %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_8]] : i64
// CHECK:             cc.condition %[[VAL_16]](%[[VAL_14]], %[[VAL_15]] : i64, i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i64, %[[VAL_18:.*]]: i32):
// CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_9]][%[[VAL_17]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:             %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i8>
// CHECK:             %[[VAL_12:.*]] = cc.cast %[[VAL_20]] : (i8) -> i1
// CHECK:             %[[VAL_21:.*]] = cc.if(%[[VAL_12]]) -> i32 {
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i32
// CHECK:               cc.continue %[[VAL_22]] : i32
// CHECK:             } else {
// CHECK:               cc.continue %[[VAL_18]] : i32
// CHECK:             }
// CHECK:             cc.continue %[[VAL_17]], %[[VAL_23:.*]] : i64, i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_24:.*]]: i64, %[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_26]], %[[VAL_25]] : i64, i32
// CHECK:           } {invariant}
// CHECK:           return %[[VAL_27:.*]]#1 : i32
// CHECK:         }

bool func_shlosh(cudaq::qvector<> &qv) __qpu__ {
  auto i = func_shiim(qv);
  return i;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_func_shlosh._Z11func_shloshRN5cudaq7qvectorILm2EEE(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) -> i1 attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = call @__nvqpp__mlirgen__function_func_shiim._Z10func_shiimRN5cudaq7qvectorILm2EEE(%[[VAL_0]]) : (!quake.veq<?>) -> i32
// CHECK:           %[[VAL_3:.*]] = arith.cmpi ne, %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_3]] : i1
// CHECK:         }

void func_arba() __qpu__ {
  cudaq::qvector qv(10);
  z(qv);
  auto b = func_shlosh(qv);
  if (b)
    h(qv);
  x(qv);
  h(qv);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_func_arba._Z9func_arbav() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 10 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_4:.*]] = quake.relax_size %[[VAL_3]] : (!quake.veq<10>) -> !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_8]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:             quake.z %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_11]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_12:.*]] = call @__nvqpp__mlirgen__function_func_shlosh._Z11func_shloshRN5cudaq7qvectorILm2EEE(%[[VAL_4]]) : (!quake.veq<?>) -> i1
// CHECK:           cc.if(%[[VAL_12]]) {
// CHECK:             %[[VAL_13:.*]] = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:               %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_0]] : i64
// CHECK:               cc.condition %[[VAL_15]](%[[VAL_14]] : i64)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_16:.*]]: i64):
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_16]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:               quake.h %[[VAL_17]] : (!quake.ref) -> ()
// CHECK:               cc.continue %[[VAL_16]] : i64
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_18:.*]]: i64):
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : i64
// CHECK:               cc.continue %[[VAL_19]] : i64
// CHECK:             } {invariant}
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = cc.loop while ((%[[VAL_21:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_22]](%[[VAL_21]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
// CHECK:             %[[VAL_24:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_23]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:             quake.x %[[VAL_24]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_23]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_26]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_27:.*]] = cc.loop while ((%[[VAL_28:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_28]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_29]](%[[VAL_28]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_30:.*]]: i64):
// CHECK:             %[[VAL_31:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_30]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_31]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_30]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_32:.*]]: i64):
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_33]] : i64
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

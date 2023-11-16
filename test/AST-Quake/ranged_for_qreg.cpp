/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void range_qubit() {
  cudaq::qreg<10> qr;

  for (auto &q : qr) {
    bool weevil = mx(q);
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_range_qubit
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : index
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_8]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:             cc.scope {
// CHECK:               %[[VAL_110:.*]] = quake.mx %[[VAL_9]] : (!quake.ref) -> !quake.measure
// CHECK:               %[[VAL_10:.*]] = quake.discriminate %[[VAL_110]] :
// CHECK:               %[[VAL_11:.*]] = cc.alloca i1
// CHECK:               cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<i1>
// CHECK:             }
// CHECK:             cc.continue %[[VAL_7]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_13]] : index
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

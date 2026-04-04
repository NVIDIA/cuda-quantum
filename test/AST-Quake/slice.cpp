/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void other(cudaq::qview<>);

struct SliceTest {
   void operator()(int i1, int i2) __qpu__ {
      cudaq::qvector reg(10);
      auto s = reg.slice(i1, i2);
      other(s);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__SliceTest
// CHECK-SAME:      (%[[VAL_0:.*]]: i32{{.*}}, %[[VAL_1:.*]]: i32{{.*}}) attributes {
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_12:.*]] = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_11]] : i64
// CHECK:           %[[VAL_14:.*]] = quake.subveq %[[VAL_6]], %{{.*}}, %[[VAL_13]] : (!quake.veq<10>, i64, i64) -> !quake.veq<?>
// CHECK:           call @{{.*}}other{{.*}}(%[[VAL_14]]) : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ bool issue_3092() {
  cudaq::qvector qubits(6);
  x(qubits[3]);
  return mz(qubits.slice(3, 1))[0];
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_issue_3092._Z10issue_3092v() -> i1
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<6>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][3] : (!quake.veq<6>) -> !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.subveq %[[VAL_1]], 3, 3 : (!quake.veq<6>) -> !quake.veq<1>
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_3]] : (!quake.veq<1>) -> !quake.measurements<1>
// CHECK:           %[[VAL_5:.*]] = quake.get_measure %[[VAL_4]][%[[VAL_0]]] : (!quake.measurements<1>, i64) -> !quake.measure
// CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_6]] : i1
// CHECK:         }

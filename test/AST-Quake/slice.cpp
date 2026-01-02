/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<6>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][3] : (!quake.veq<6>) -> !quake.ref
// CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.subveq %[[VAL_0]], 3, 3 : (!quake.veq<6>) -> !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[VAL_3]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_4]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (i8) -> i1
// CHECK:           return %[[VAL_8]] : i1
// CHECK:         }

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__SliceTest(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                           %[[VAL_1:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.cast signed %[[VAL_8]] : (i32) -> i64
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_12:.*]] = quake.subveq %[[VAL_5]], %[[VAL_7]], %[[VAL_11]] : (!quake.veq<10>, i64, i64) -> !quake.veq<?>
// CHECK:           call @{{.*}}other{{.*}}(%[[VAL_12]]) : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ bool issue_3092() {
  cudaq::qvector qubits(6);
  x(qubits[3]);
  return mz(qubits.slice(3, 1))[0];
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_issue_3092._Z10issue_3092v() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<6>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][3] : (!quake.veq<6>) -> !quake.ref
// CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.subveq %[[VAL_0]], 3, 3 : (!quake.veq<6>) -> !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<!quake.measure x ?>>) -> !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_7:.*]] = quake.discriminate %[[VAL_6]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_7]] : i1
// CHECK:         }

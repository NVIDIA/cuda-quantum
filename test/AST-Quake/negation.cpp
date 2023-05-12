/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct NegationOperatorTest {
  void operator()() __qpu__ {
    cudaq::qreg qr(3);
    x<cudaq::ctrl>(!qr[0], qr[1], qr[2]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__NegationOperatorTest
// CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca[%[[VAL_1]] : i64] !quake.qvec<?>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.extsi %[[VAL_3]] : i32 to i64
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_4]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_7]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_10:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_10]]] : (!quake.qvec<?>, i64) -> !quake.ref
// CHECK:           quake.x [%[[VAL_5]], %[[VAL_8]] neg [true, false]] %[[VAL_11]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }



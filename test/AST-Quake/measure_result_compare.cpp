/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ int compare_kernel() {
  cudaq::qvector q(2);
  cudaq::measure_result a = mz(q[0]);
  cudaq::measure_result b = mz(q[1]);
  if (a == b)
    return 1;
  if (a != b)
    return 0;
  return -1;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_compare_kernel._Z14compare_kernelv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_4]] name "a" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_6:.*]] = cc.alloca !quake.measure
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_7]] name "b" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_9:.*]] = cc.alloca !quake.measure
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_6]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_9]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_12:.*]] = quake.discriminate %[[VAL_10]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_13:.*]] = quake.discriminate %[[VAL_11]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_13]] : i1
// CHECK:           cc.if(%[[VAL_14]]) {
// CHECK:             cc.unwind_return %[[VAL_1]] : i32
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_6]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_9]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_17:.*]] = quake.discriminate %[[VAL_15]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_18:.*]] = quake.discriminate %[[VAL_16]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_18]] : i1
// CHECK:           cc.if(%[[VAL_19]]) {
// CHECK:             cc.unwind_return %[[VAL_2]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
// clang-format on

__qpu__ int compare_with_bool_kernel() {
  cudaq::qubit q;
  cudaq::measure_result a = mz(q);
  if (a == true)
    return 1;
  if (a != false)
    return 2;
  return 0;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_compare_with_bool_kernel._Z24compare_with_bool_kernelv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] name "a" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_7:.*]] = cc.alloca !quake.measure
// CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_9:.*]] = quake.discriminate %[[VAL_8]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:           cc.if(%[[VAL_10]]) {
// CHECK:             cc.unwind_return %[[VAL_3]] : i32
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_7]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_12:.*]] = quake.discriminate %[[VAL_11]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_2]] : i1
// CHECK:           cc.if(%[[VAL_13]]) {
// CHECK:             cc.unwind_return %[[VAL_1]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
// clang-format on

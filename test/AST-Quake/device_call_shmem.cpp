/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --device-call-shmem | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --apply-op-specialization | FileCheck --check-prefix=ADJOINT %s

#include <cudaq.h>

int bar(bool a);

__qpu__ auto foo() {
  cudaq::qubit q;
  h(q);
  auto bit = mz(q);
  auto result = cudaq::device_call(bar, bit);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo._Z3foov() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "bit" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_3:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = cc.string_literal "_Z3barb" : !cc.ptr<!cc.array<i8 x 8>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<i8 x 8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.array<!cc.ptr<i8> x 1>
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.array<i64 x 1>
// CHECK:           %[[VAL_11:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_4]], %[[VAL_11]] : !cc.ptr<i1>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<!cc.ptr<i8> x 1>>, i64) -> !cc.ptr<!cc.ptr<i8>>
// CHECK:           cc.store %[[VAL_12]], %[[VAL_14]] : !cc.ptr<!cc.ptr<i8>>
// CHECK:           %[[VAL_15:.*]] = cc.sizeof i1 : i64
// CHECK:           %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<i64 x 1>>, i64) -> !cc.ptr<i64>
// CHECK:           cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = cc.alloca i32
// CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<i32>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = cc.sizeof i32 : i64
// CHECK:           %[[VAL_20:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<!cc.ptr<i8> x 1>>) -> !cc.ptr<!cc.ptr<i8>>
// CHECK:           %[[VAL_21:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.array<i64 x 1>>) -> !cc.ptr<i64>
// CHECK:           call @__nvqlink_device_call_dispatch(%[[VAL_5]], %[[VAL_7]], %[[VAL_8]], %[[VAL_20]], %[[VAL_21]], %[[VAL_18]], %[[VAL_19]]) : (i64, !cc.ptr<i8>, i64, !cc.ptr<!cc.ptr<i8>>, !cc.ptr<i64>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_22:.*]] = cc.load %[[VAL_17]] : !cc.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_22]], %[[VAL_23]] : !cc.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_24]] : i32
// CHECK:         }
// CHECK:         func.func private @_Z3barb(i1) -> i32 attributes {"cudaq-devicecall"}
// clang-format on

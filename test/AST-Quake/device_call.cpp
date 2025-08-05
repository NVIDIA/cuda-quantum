/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

int fun1(int a, int b);
void fun2(int *a, int *b);
int fun3();

__qpu__ auto sufur() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call(fun1, 1, 2);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_sufur._Z5sufurv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.device_call @_Z4fun1ii(%[[VAL_5]], %[[VAL_6]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_8:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_9]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto bir(cudaq::device_ptr<int> p1, cudaq::device_ptr<int> p2) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call(fun2, p1, p2);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_bir._Z3birN5cudaq10device_ptrIiEES1_(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>,
// CHECK-SAME:      %[[VAL_1:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = cc.resolve_device_ptr %[[VAL_2]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.resolve_device_ptr %[[VAL_3]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           cc.device_call @_Z4fun2PiS_(%[[VAL_5]], %[[VAL_6]]) : (!cc.ptr<i32>, !cc.ptr<i32>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ auto iki() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call(fun3);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_iki._Z3ikiv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = cc.device_call @_Z4fun3v() : () -> i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto uech() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call(1, fun1, 1, 2);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_uech._Z4uechv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.device_call @_Z4fun1ii on %[[VAL_0]](%[[VAL_6]], %[[VAL_7]]) : (i64, i32, i32) -> i32
// CHECK:           %[[VAL_9:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_10]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto doert(cudaq::device_ptr<int> p1, cudaq::device_ptr<int> p2) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call(2, fun2, p1, p2);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_doert._Z5doertN5cudaq10device_ptrIiEES1_(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>,
// CHECK-SAME:      %[[VAL_1:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_6:.*]] = cc.resolve_device_ptr %[[VAL_3]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.resolve_device_ptr %[[VAL_4]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           cc.device_call @_Z4fun2PiS_ on %[[VAL_2]](%[[VAL_6]], %[[VAL_7]]) : (i64, !cc.ptr<i32>, !cc.ptr<i32>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ auto besh() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call(fun3);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_besh._Z4beshv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = cc.device_call @_Z4fun3v() : () -> i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto altu() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call<256, 128>(fun1, 1, 2);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_altu._Z4altuv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.device_call @_Z4fun1ii<%[[VAL_1]] * %[[VAL_0]]>(%[[VAL_7]], %[[VAL_8]]) : (i64, i64, i32, i32) -> i32
// CHECK:           %[[VAL_10:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto yedi(cudaq::device_ptr<int> p1, cudaq::device_ptr<int> p2) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call<256, 128>(fun2, p1, p2);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_yedi._Z4yediN5cudaq10device_ptrIiEES1_(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>,
// CHECK-SAME:      %[[VAL_1:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_6]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = cc.resolve_device_ptr %[[VAL_4]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.resolve_device_ptr %[[VAL_5]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           cc.device_call @_Z4fun2PiS_<%[[VAL_3]] * %[[VAL_2]]>(%[[VAL_7]], %[[VAL_8]]) : (i64, i64, !cc.ptr<i32>, !cc.ptr<i32>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ auto sekiz() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call<256, 128>(fun3);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_sekiz._Z5sekizv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = cc.device_call @_Z4fun3v<%[[VAL_1]] * %[[VAL_0]]>() : (i64, i64) -> i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto dokuz() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call<256, 128>(4, fun1, 1, 2);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_dokuz._Z5dokuzv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.device_call @_Z4fun1ii<%[[VAL_2]] * %[[VAL_1]]> on %[[VAL_0]](%[[VAL_8]], %[[VAL_9]]) : (i64, i64, i64, i32, i32) -> i32
// CHECK:           %[[VAL_11:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_12]] : i32
// CHECK:         }
// clang-format on

__qpu__ auto on(cudaq::device_ptr<int> p1, cudaq::device_ptr<int> p2) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call<256, 128>(5, fun2, p1, p2);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_on._Z2onN5cudaq10device_ptrIiEES1_(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>,
// CHECK-SAME:      %[[VAL_1:.*]]: !cc.struct<"device_ptr" {i64, i64, i64}>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 5 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64}>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_7]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_8:.*]] = cc.resolve_device_ptr %[[VAL_5]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.resolve_device_ptr %[[VAL_6]] : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i32>
// CHECK:           cc.device_call @_Z4fun2PiS_<%[[VAL_4]] * %[[VAL_3]]> on %[[VAL_2]](%[[VAL_8]], %[[VAL_9]]) : (i64, i64, i64, !cc.ptr<i32>, !cc.ptr<i32>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ auto on_bir() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call<256, 128>(6, fun3);
  return result;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_on_bir._Z6on_birv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 6 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 128 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 256 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = cc.device_call @_Z4fun3v<%[[VAL_2]] * %[[VAL_1]]> on %[[VAL_0]]() : (i64, i64, i64) -> i32
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }

// CHECK:         func.func private @_Z4fun1ii(i32, i32) -> i32 attributes {"cudaq-devicecall"}

// CHECK:         func.func private @_Z4fun2PiS_(!cc.ptr<i32>, !cc.ptr<i32>) attributes {"cudaq-devicecall"}

// CHECK:         func.func private @_Z4fun3v() -> i32 attributes {"cudaq-devicecall"}
// clang-format on

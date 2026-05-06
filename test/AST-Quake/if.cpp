/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// Test the lowering of if and if-else statements.

struct kernel {
  __qpu__ int operator()(bool flag) {
    cudaq::qvector reg(2);
    if (flag) {
      h<cudaq::ctrl>(reg[0], reg[1]);
    }
    return 0;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
// CHECK-SAME:      %[[VAL_0:.*]]: i1{{.*}}) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.h [%[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32

struct kernel_else {
  __qpu__ int operator()(bool flag) {
    cudaq::qvector reg(2);
    if (flag) {
      h<cudaq::ctrl>(reg[0], reg[1]);
    } else {
      x<cudaq::ctrl>(reg[1], reg[0]);
    }
    return 0;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_else(
// CHECK-SAME:      %[[VAL_0:.*]]: i1{{.*}}) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.h {{\[}}%[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x {{\[}}%[[VAL_7]]] %[[VAL_8]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32

struct kernel_short_circuit_and {
  __qpu__ int operator()() {
    cudaq::qvector reg(3);
    if (mz(reg[0]) && mz(reg[1]))
      x(reg[2]);
    return 0;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_short_circuit_and() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_10]], %[[HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[HL:.*]] = cc.load %[[HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[HL]] :
// CHECK:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_0]] : i1
// CHECK:           %[[VAL_6:.*]] = cc.if(%[[VAL_5]]) -> i1 {
// CHECK:             cc.continue %[[VAL_0]] : i1
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:             %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[HA2:.*]] = cc.alloca !cc.measure_handle
// CHECK:             cc.store %[[VAL_8]], %[[HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[HL2:.*]] = cc.load %[[HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_81:.*]] = quake.discriminate %[[HL2]] :
// CHECK:             cc.continue %[[VAL_81]] : i1
// CHECK:           }
// CHECK:           cc.if(%[[VAL_6]]) {
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_2]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }

struct kernel_short_circuit_or {
  __qpu__ int operator()() {
    cudaq::qvector reg(3);
    if (mz(reg[0]) || mz(reg[1]))
      x(reg[2]);
    return 0;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_short_circuit_or() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_41:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[HAo:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_41]], %[[HAo]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[HLo:.*]] = cc.load %[[HAo]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[HLo]] :
// CHECK:           %[[VAL_6:.*]] = cc.if(%[[VAL_4]]) -> i1 {
// CHECK:             cc.continue %[[VAL_4]] : i1
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:             %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[HA2o:.*]] = cc.alloca !cc.measure_handle
// CHECK:             cc.store %[[VAL_8]], %[[HA2o]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[HL2o:.*]] = cc.load %[[HA2o]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_81:.*]] = quake.discriminate %[[HL2o]] :
// CHECK:             cc.continue %[[VAL_81]] : i1
// CHECK:           }
// CHECK:           cc.if(%[[VAL_6]]) {
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_2]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }

struct kernel_ternary {
  __qpu__ int operator()() {
    cudaq::qvector q(3);
    auto measureResult = mz(q[0]) ? mz(q[1]) : mz(q[2]);
    return 0;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_ternary() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[HAt:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_3]], %[[HAt]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[HLt:.*]] = cc.load %[[HAt]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_31:.*]] = quake.discriminate %[[HLt]] :
// CHECK:           %[[VAL_4:.*]] = cc.if(%[[VAL_31]]) -> !cc.measure_handle {
// CHECK:             %[[VAL_5:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:             %[[VAL_6:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> !cc.measure_handle
// CHECK:             cc.continue %[[VAL_6]] : !cc.measure_handle
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_1]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:             %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !cc.measure_handle
// CHECK:             cc.continue %[[VAL_8]] : !cc.measure_handle
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_4]], %[[VAL_9]] : !cc.ptr<!cc.measure_handle>
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

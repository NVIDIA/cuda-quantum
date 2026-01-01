/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct testCast {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    double bit = mz(q0);
    // This tests implicit casting from double to bool
    if (bit)
      x(q1);
    mz(q1);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testCast() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_10:.*]] = quake.discriminate %[[VAL_3]] :
// CHECK:           %[[VAL_4:.*]] = cc.cast unsigned %[[VAL_10]] : (i1) -> f64
// CHECK:           %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = arith.cmpf une, %[[VAL_6]], %[[VAL_0]] : f64
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct testCastBoolMeasurement {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    bool bit = mz(q0);
    // This tests casting from bool to uint32
    unsigned int i = (unsigned)(bit);
    if (i == 1)
      x(q1);
    mz(q1);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testCastBoolMeasurement() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1_i32:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_m0:.*]] = quake.mz %[[VAL_0]] name {{.*}} : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_m0]] :
// CHECK:           %[[VAL_3:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.cast unsigned %[[VAL_4]] : (i1) -> i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7:.*]], %[[VAL_1_i32]] : i32
// CHECK:           cc.if(%[[VAL_8]]) {
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_m1:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct testUnsignedCastBoolConstTrue {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    // This tests casting from bool to uint32
    // and constant folding
    unsigned i = (unsigned)(true);
    if (i == 1)
      x(q0);
    mz(q0);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testUnsignedCastBoolConstTrue() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1_i32:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1_i32]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_2:.*]], %[[VAL_1_i32]] : i32
// CHECK:           cc.if(%[[VAL_3]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_m:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct testUnsignedCastBoolConstFalse {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    // This tests casting from bool to uint32
    // and constant folding
    unsigned i = (unsigned)(false);
    if (i == 0)
      x(q0);
    mz(q0);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testUnsignedCastBoolConstFalse() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0_i32:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0_i32]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_2:.*]], %[[VAL_0_i32]] : i32
// CHECK:           cc.if(%[[VAL_3]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_m:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct testSignedCastBoolConstTrue {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    // This tests casting from bool to int32
    // and constant folding
    signed int i = (signed)(true);
    if (i == 1)
      x(q0);
    mz(q0);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testSignedCastBoolConstTrue() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1_i32:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1_i32]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_2:.*]], %[[VAL_1_i32]] : i32
// CHECK:           cc.if(%[[VAL_3]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_m:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

struct testSignedCastBoolConstFalse {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    // This tests casting from bool to int32
    // and constant folding.
    signed int i = (signed)(false);
    if (i == 0)
      x(q0);
    mz(q0);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testSignedCastBoolConstFalse() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0_i32:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0_i32]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_2:.*]], %[[VAL_0_i32]] : i32
// CHECK:           cc.if(%[[VAL_3]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_m:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

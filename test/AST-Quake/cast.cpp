/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct testCast {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    double bit = static_cast<double>(mz(q0));
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
// CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[VAL_3]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_5:.*]] = cc.cast unsigned %[[VAL_4]] : (i1) -> f64
// CHECK:           %[[VAL_6:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = arith.cmpf une, %[[VAL_7]], %[[VAL_0]] : f64
// CHECK:           cc.if(%[[VAL_8]]) {
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[VAL_3]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_5:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i1>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i1>
// CHECK:           %[[VAL_7:.*]] = cc.cast unsigned %[[VAL_6]] : (i1) -> i32
// CHECK:           %[[VAL_8:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_10]]) {
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

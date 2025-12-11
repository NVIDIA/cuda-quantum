/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck --check-prefixes=CHECK,ALIVE %s
// RUN: cudaq-quake %cpp_std %s | cudaq-opt -erase-detectors | FileCheck --check-prefixes=CHECK,DEAD %s
// clang-format on

#include <cudaq.h>

struct testDetector {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    mz(q0);
    mz(q1);
    cudaq::detector(-1, -2);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testDetector() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// ALIVE:           %[[VAL_0:.*]] = arith.constant -2 : i64
// ALIVE:           %[[VAL_1:.*]] = arith.constant -1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !quake.measure
// ALIVE:           "quake.detector"(%[[VAL_1]], %[[VAL_0]]) : (i64, i64) -> ()
// DEAD-NOT:       quake.detector
// CHECK:           return
// CHECK:         }

// clang-format on

struct testDetectorVec {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    mz(q0);
    mz(q1);
    cudaq::detector(std::vector<std::int64_t>{-1, -2});
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testDetectorVec() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-ALIVE:     %[[VAL_0:.*]] = arith.constant -2 : i64
// CHECK-ALIVE:     %[[VAL_1:.*]] = arith.constant -1 : i64
// CHECK-ALIVE:     %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> !quake.measure
// CHECK-ALIVE:     %[[VAL_7:.*]] = cc.alloca !cc.array<i64 x 2>
// CHECK-ALIVE:     %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<!cc.array<i64 x ?>>
// CHECK-ALIVE:     %[[VAL_9:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
// CHECK-ALIVE:     cc.store %[[VAL_1]], %[[VAL_9]] : !cc.ptr<i64>
// CHECK-ALIVE:     %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
// CHECK-ALIVE:     cc.store %[[VAL_0]], %[[VAL_10]] : !cc.ptr<i64>
// CHECK-ALIVE:     %[[VAL_11:.*]] = cc.stdvec_init %[[VAL_8]], %[[VAL_2]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.stdvec<i64>
// CHECK-ALIVE:     "quake.detector"(%[[VAL_11]]) : (!cc.stdvec<i64>) -> ()
// CHECK-DEAD-NOT:  quake.detector
// CHECK:           return
// CHECK:         }
// clang-format on

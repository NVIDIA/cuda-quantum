/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simple test to make sure the tool is built and has basic functionality.

// RUN: cudaq-quake --emit-llvm-file %s | cudaq-opt | FileCheck %s && FileCheck --check-prefix=LLVM %s < simple.ll

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
struct ghz {
  auto operator()(const int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.cast signed %[[VAL_4]] : (i32) -> i64
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>[%[[VAL_5]] : i64]
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.h %[[VAL_7]] : (!quake.ref) -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_8:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_11]] : i32
// CHECK:               cc.condition %[[VAL_12]]
// CHECK:             } do {
// CHECK:               %[[VAL_13:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_14:.*]] = cc.cast signed %[[VAL_13]] : (i32) -> i64
// CHECK:               %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_18:.*]] = cc.cast signed %[[VAL_17]] : (i32) -> i64
// CHECK:               %[[VAL_19:.*]] = quake.extract_ref %[[VAL_6]][%[[VAL_18]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.x [%[[VAL_15]]] %[[VAL_19]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_20:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_21]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = quake.mz %[[VAL_6]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

int main() {
  // Run the kernel in NISQ mode (i.e. run and
  // collect bit strings and counts)
  auto counts = cudaq::sample(ghz{}, 30);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", counts.expectation());

  return 0;
}

// LLVM: define {{(dso_local )?}}noundef i32 @main

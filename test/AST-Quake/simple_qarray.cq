/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simple test to make sure the tool is built and has basic functionality.

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck %s
// RUN: FileCheck --check-prefixes=CHECK-LLVM %s < simple_qarray.ll

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
struct ghz {
  auto operator()() __qpu__ {
    cudaq::qreg<5> q;
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

// CHECK-LLVM: define {{(dso_local )?}}noundef i32 @main

int main() {
  // Run the kernel in NISQ mode (i.e. run and
  // collect bit strings and counts)
  auto counts = cudaq::sample(ghz{});
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", counts.exp_val_z());

  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<5>
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<5>) -> !quake.ref
// CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_5:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_1]] : i32
// CHECK:               cc.condition %[[VAL_7]]
// CHECK:             } do {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : i32 to i64
// CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_9]]] : (!quake.veq<5>, i64) -> !quake.ref
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_0]] : i32
// CHECK:               %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : i32 to i64
// CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_13]]] : (!quake.veq<5>, i64) -> !quake.ref
// CHECK:               quake.x [%[[VAL_10]]] %[[VAL_14]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_15:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_0]] : i32
// CHECK:               cc.store %[[VAL_16]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = quake.mz %[[VAL_3]] : (!quake.veq<5>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }


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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz
// CHECK-SAME: ()
// CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_3:.*]] = quake.alloca[%[[VAL_2]] : i64] !quake.qvec<5>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.qvec<5>, i64) -> !quake.ref
// CHECK:           quake.h %[[VAL_8]] :
// CHECK:           cc.scope {
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_10:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_9]], %[[VAL_10]][] : memref<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_11:.*]] = memref.load %[[VAL_10]][] : memref<i32>
// CHECK:               %[[VAL_12:.*]] = arith.constant 4 : i32
// CHECK:               %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               cc.condition %[[VAL_13]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_14:.*]] = memref.load %[[VAL_10]][] : memref<i32>
// CHECK:                 %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : i32 to i64
// CHECK:                 %[[VAL_16:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_15]]] : (!quake.qvec<5>, i64) -> !quake.ref
// CHECK:                 %[[VAL_17:.*]] = memref.load %[[VAL_10]][] : memref<i32>
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:                 %[[VAL_20:.*]] = arith.extsi %[[VAL_19]] : i32 to i64
// CHECK:                 %[[VAL_21:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_20]]] : (!quake.qvec<5>, i64) -> !quake.ref
// CHECK:                 quake.x [%[[VAL_16]]] %[[VAL_21]] :
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_10]][] : memref<i32>
// CHECK:               %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:               memref.store %[[VAL_24]], %[[VAL_10]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = quake.mz %[[VAL_3]] : (!quake.qvec<5>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }


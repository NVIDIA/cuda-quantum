/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simple test to make sure the tool is built and has basic functionality.

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck --check-prefixes=CHECK %s
// RUN: FileCheck --check-prefixes=CHECK-LLVM %s < simple.ll

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz
// CHECK-SAME:        (%[[VAL_0:.*]]: i32{{.*}})
// CHECK-VISIT:     %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK-VISIT:     memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK-VISIT:     %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK-VISIT:     %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK-VISIT:     %[[VAL_4:.*]] = quake.alloca(%[[VAL_3]] : i64) : !quake.veq<?>
// CHECK-VISIT:     %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-VISIT:     %[[VAL_6:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK-VISIT:     %[[VAL_7:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_6]] : i64] : !quake.veq<?> -> !quake.ref
// CHECK-VISIT:     quake.h (%[[VAL_7]])
// CHECK-VISIT:     quake.scope {
// CHECK-VISIT:       %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK-VISIT:       %[[VAL_9:.*]] = memref.alloca() : memref<i32>
// CHECK-VISIT:       memref.store %[[VAL_8]], %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:       quake.loop while {
// CHECK-VISIT:         %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:         %[[VAL_11:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK-VISIT:         %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK-VISIT:         %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK-VISIT:         %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_13]] : i32
// CHECK-VISIT:         quake.condition %[[VAL_14]] ()
// CHECK-VISIT:       } do {
// CHECK-VISIT:         quake.scope {
// CHECK-VISIT:           %[[VAL_15:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:           %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK-VISIT:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_16]] : i64] : !quake.veq<?> -> !quake.ref
// CHECK-VISIT:           %[[VAL_18:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:           %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK-VISIT:           %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] : i32
// CHECK-VISIT:           %[[VAL_21:.*]] = arith.extsi %[[VAL_20]] : i32 to i64
// CHECK-VISIT:           %[[VAL_22:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_21]] : i64] : !quake.veq<?> -> !quake.ref
// CHECK-VISIT:           quake.x [%[[VAL_17]]] (%[[VAL_22]])
// CHECK-VISIT:         }
// CHECK-VISIT:         quake.continue ()
// CHECK-VISIT:       } step {
// CHECK-VISIT:         %[[VAL_23:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:         %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK-VISIT:         %[[VAL_25:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i32
// CHECK-VISIT:         memref.store %[[VAL_25]], %[[VAL_9]][] : memref<i32>
// CHECK-VISIT:       }
// CHECK-VISIT:     }
// CHECK-VISIT:     %[[VAL_26:.*]] = quake.veq_size(%[[VAL_4]] : !quake.veq<?>) : i64
// CHECK-VISIT:     %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : i64 to index
// CHECK-VISIT:     %[[VAL_28:.*]] = arith.constant 0 : index
// CHECK-VISIT:     affine.for %[[VAL_29:.*]] = affine_map<(d0) -> (d0)>(%[[VAL_28]]) to affine_map<(d0) -> (d0)>(%[[VAL_27]]) {
// CHECK-VISIT:       %[[VAL_30:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_29]] : index] : !quake.veq<?> -> !quake.ref
// CHECK-VISIT:       %[[VAL_31:.*]] = quake.mz(%[[VAL_30]] : !quake.ref) : i1
// CHECK-VISIT:     }
// CHECK-VISIT:     return
// CHECK-VISIT:   }

// CHECK-LLVM: define {{(dso_local )?}}noundef i32 @main

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
struct ghz {
  auto operator()(const int N) __qpu__ {
    cudaq::qreg q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

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
  printf("Exp: %lf\n", counts.exp_val_z());

  return 0;
}

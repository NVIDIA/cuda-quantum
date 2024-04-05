/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt --memtoreg=quantum=0 --cc-loop-peeling --canonicalize --cc-loop-normalize --cse --cc-loop-unroll | FileCheck %s
// clang-format on

#include <cudaq.h>

// This example demonstrates a combination of loop peeling (converting the
// do-while loop to a peeled iteration + a while loop), followed by loop
// normalization (to turn the residual while loop into a counted loop of 9
// iterations), followed by fully unrolling the counted loop (9x). This yields
// exactly 10 measurements on $[0 \dots 9]$.

__qpu__ void loop_peeling_and_unrolling_test() {
  cudaq::qvector r(10);
  unsigned i = 0;

  do {
    mz(r[i]);
    i++;
  } while (i < 10);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_loop_peeling_and_unrolling_test
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_0]][3] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]][4] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_9]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_0]][5] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_11]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_0]][6] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_14:.*]] = quake.mz %[[VAL_13]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_0]][7] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_16:.*]] = quake.mz %[[VAL_15]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]][8] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_18:.*]] = quake.mz %[[VAL_17]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_19:.*]] = quake.extract_ref %[[VAL_0]][9] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_20:.*]] = quake.mz %[[VAL_19]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void another_test() {
  cudaq::qvector r(10);
  unsigned i = 0;

  do {
    mz(r[i]);
  } while (i++ < 9);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_another_test
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<10>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_0]][3] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]][4] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_9]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_0]][5] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_11]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_0]][6] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_14:.*]] = quake.mz %[[VAL_13]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_0]][7] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_16:.*]] = quake.mz %[[VAL_15]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]][8] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_18:.*]] = quake.mz %[[VAL_17]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_19:.*]] = quake.extract_ref %[[VAL_0]][9] : (!quake.veq<10>) -> !quake.ref
// CHECK:           %[[VAL_20:.*]] = quake.mz %[[VAL_19]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }

struct Qernel {
  // Loop that decrements. Loop is not unrolled. It needs to be normalized.
  void operator()() __qpu__ {
    cudaq::qvector reg(1);
    for (size_t i = 3; i-- > 0;)
      x(reg[0]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel()
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

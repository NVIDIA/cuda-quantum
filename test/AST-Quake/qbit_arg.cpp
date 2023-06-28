/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck %s

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_testFunc
// CHECK-SAME:    (%[[VAL_0:.*]]: !quake.ref{{.*}})
// CHECK: quake.h %[[VAL_0]] :
// CHECK: %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> i1
// CHECK: return
// CHECK: }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Test
// CHECK-SAME: () attributes
// CHECK: %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK: call @__nvqpp__mlirgen__function_testFunc{{.*}}(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK: return
// CHECK: }

#include <cudaq.h>
#include <utility>

using namespace cudaq;

void testFunc(qubit &q) __qpu__ {
  h(q);
  mz(q);
}

struct Test {
  void operator()() __qpu__ {
    cudaq::qubit q;
    testFunc(q);
  }
};

int main() {
  auto counts = cudaq::sample(Test{});
  counts.dump();
}

/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck %s

#include <cudaq.h>
#include <utility>

using namespace cudaq;

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Teste
// CHECK-SAME: (%[[VAL_0:.*]]: !quake.ref{{.*}})
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> i1
// CHECK:           return
// CHECK:         }

struct Teste {
  void operator()(qubit &q) __qpu__ {
    h(q);
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Test
// CHECK-SAME: ()
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           call @__nvqpp__mlirgen__Teste{{.*}}(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

struct Test {
  void operator()() __qpu__ {
    cudaq::qubit q;
    Teste{}(q);
  }
};

int main() {
  auto counts = cudaq::sample(Test{});
  counts.dump();
}

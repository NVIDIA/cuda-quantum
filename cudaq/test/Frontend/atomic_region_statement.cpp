/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// clang-format off

#include <cudaq.h>

// A compound statement with the atomic quantum region attribute is a `cc.scope`
// that has the `atomic_quantum_region` attribute. The attribute is spelled
// `[[cudaq::atomic_region]]`, and the annotation that it stands for works too.
struct atomic_block {
  void operator()() __qpu__ {
    cudaq::qubit q;
    [[cudaq::atomic_region]] {
      x(q);
      h(q);
    }
    y(q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__atomic_block()
// CHECK:         %[[Q:.*]] = quake.alloca !quake.ref
// CHECK:         cc.scope atomic {
// CHECK:           quake.x %[[Q]] : (!quake.ref) -> ()
// CHECK:           quake.h %[[Q]] : (!quake.ref) -> ()
// CHECK:         }
// CHECK:         quake.y %[[Q]] : (!quake.ref) -> ()

struct annotated_block {
  void operator()() __qpu__ {
    cudaq::qubit q;
    [[clang::annotate("atomic_quantum_region")]] { x(q); }
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__annotated_block()
// CHECK:         cc.scope atomic {
// CHECK:           quake.x
// CHECK:         }

// On a function, it is the same as `__atomic_quantum_region__`.
[[cudaq::atomic_region]] __qpu__ void atomic_function(cudaq::qubit &q) { h(q); }

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_atomic_function.
// CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-kernel", no_this}

// Other attributes on statements do not change the code.
struct other_attributes {
  void operator()(int n) __qpu__ {
    cudaq::qubit q;
    if (n > 1) [[likely]] { x(q); } else [[unlikely]] { h(q); }
    [[clang::annotate("something_else")]] { y(q); }
    [[maybe_unused]] int unused = 3;
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__other_attributes(
// CHECK-NOT:     cc.scope atomic
// CHECK:         return
// clang-format on

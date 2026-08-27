/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

__qpu__ __atomic_quantum_region__ void atomic_function(cudaq::qubit &q) {
  h(q);
}

__qpu__ void ordinary_function(cudaq::qubit &q) { h(q); }

struct atomic_functor {
  void operator()(cudaq::qubit &q) __qpu__ __atomic_quantum_region__ { x(q); }
};

struct atomic_lambda {
  void operator()() __qpu__ {
    cudaq::qarray<2> q;
    auto callable = [](cudaq::qubit &target)
                        __qpu__ __atomic_quantum_region__ { y(target); };
    cudaq::control(callable, q[0], q[1]);
  }
};

// clang-format off
// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_atomic_function.
// CHECK-SAME:    %[[ARG0:.*]]: !quake.ref{{.*}} attributes {atomic_quantum_region, "cudaq-kernel", no_this}
// CHECK:         quake.h %[[ARG0]] : (!quake.ref) -> ()

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_ordinary_function.
// CHECK-SAME:    %[[ARG0:.*]]: !quake.ref{{.*}} attributes {"cudaq-kernel", no_this}
// CHECK:         quake.h %[[ARG0]] : (!quake.ref) -> ()

// CHECK-LABEL: func.func @__nvqpp__mlirgen__atomic_functor(
// CHECK-SAME:    %[[ARG0:.*]]: !quake.ref{{.*}} attributes {atomic_quantum_region, "cudaq-kernel"}
// CHECK:         quake.x %[[ARG0]] : (!quake.ref) -> ()

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ZN13atomic_lambda
// CHECK-SAME:    (%[[ARG0:.*]]: !quake.ref{{.*}}) attributes {atomic_quantum_region, "cudaq-kernel"}
// CHECK:         quake.y %[[ARG0]] : (!quake.ref) -> ()
// clang-format on

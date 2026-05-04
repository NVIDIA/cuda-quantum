/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Verify adjoint and controlled modifiers can be combined in two ways:
//   1. control(adj(k)): wrap adjoint(k) in a `__qpu__` struct, then control it.
//   2. adj(control(k)): wrap control(k) in a `__qpu__` struct, then adjoint it.
// (https://github.com/NVIDIA/cuda-quantum/issues/854)

#include <cudaq.h>

struct k {
  void operator()(cudaq::qview<> q) __qpu__ {
    h(q[0]);
    t(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__k(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>
// CHECK:           quake.h %{{.*}}
// CHECK:           quake.t %{{.*}}
// CHECK:           return
// clang-format on

struct k_adj {
  void operator()(cudaq::qview<> q) __qpu__ { cudaq::adjoint(k{}, q); }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__k_adj(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>
// CHECK:           quake.apply<adj> @__nvqpp__mlirgen__k %[[VAL_0]]
// CHECK:           return
// clang-format on

struct ep {
  void operator()() __qpu__ {
    cudaq::qarray<2> q;
    cudaq::qubit ctrl;
    cudaq::control(k_adj{}, {ctrl}, q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ep()
// CHECK:           %[[CTRL:.*]] = quake.alloca !quake.ref
// CHECK:           %[[Q:.*]] = quake.relax_size %{{.*}} : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           quake.apply @__nvqpp__mlirgen__k_adj {{\[}}%[[CTRL]]] %[[Q]]
// CHECK:           return
// clang-format on

// Approach 2: adj(control(k)) -- wrap control(k) in a `__qpu__` struct, adjoint
// it.
struct k_ctrl {
  void operator()(cudaq::qubit &ctrl, cudaq::qview<> q) __qpu__ {
    cudaq::control(k{}, {ctrl}, q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__k_ctrl(
// CHECK-SAME:      %[[CTRL:.*]]: !quake.ref
// CHECK-SAME:      %[[Q:.*]]: !quake.veq<?>
// CHECK:           quake.apply @__nvqpp__mlirgen__k {{\[}}%[[CTRL]]] %[[Q]]
// CHECK:           return
// clang-format on

struct ep2 {
  void operator()() __qpu__ {
    cudaq::qarray<2> q;
    cudaq::qubit ctrl;
    cudaq::adjoint(k_ctrl{}, ctrl, q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ep2()
// CHECK:           %[[CTRL2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[Q2:.*]] = quake.relax_size %{{.*}} : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           quake.apply<adj> @__nvqpp__mlirgen__k_ctrl %[[CTRL2]], %[[Q2]]
// CHECK:           return
// clang-format on

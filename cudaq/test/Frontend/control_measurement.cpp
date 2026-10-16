/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --apply-op-specialization --canonicalize | FileCheck %s
// clang-format on

// cudaq::control over a measured-uncomputation block: the kernel allocates
// its scratch qubit, the compute Toffoli and the payload receive the control,
// the X-basis measurement does not, the outcome-conditioned feedforward CZ
// receives the control, and the reset of the measured qubit does not. The C++
// bridge spills the measurement handle through a local; memtoreg (as in the
// default pipelines) collapses that before specialization.

#include <cudaq.h>

struct mbu {
  void operator()(cudaq::qview<> q) __qpu__ {
    cudaq::qubit anc;
    x<cudaq::ctrl>(q[0], q[1], anc);
    z(anc);
    auto fired = mx(anc);
    if (fired) {
      z<cudaq::ctrl>(q[0], q[1]);
      x(anc);
    }
  }
};

struct driver {
  void operator()() __qpu__ {
    cudaq::qvector c(1);
    cudaq::qvector q(2);
    ry(0.7, c[0]);
    h(q[0]);
    h(q[1]);
    cudaq::control(mbu{}, c[0], q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__mbu.ctrl(
// CHECK-SAME:      %[[CTRL:.*]]: !quake.veq<?>, %[[Q:.*]]: !quake.veq<?>) {
// CHECK:           %[[ANC:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x {{\[}}%[[CTRL]], %{{.*}}, %{{.*}}] %[[ANC]] : (!quake.veq<?>, !quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           quake.z {{\[}}%[[CTRL]]] %[[ANC]] : (!quake.veq<?>, !quake.ref) -> ()
// CHECK:           %[[M:.*]] = quake.mx %[[ANC]] name "fired" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[BIT:.*]] = quake.discriminate %[[M]] : (!cc.measure_handle) -> i1
// CHECK:           cc.if(%[[BIT]]) {
// CHECK:             quake.z {{\[}}%[[CTRL]], %{{.*}}] %{{.*}} : (!quake.veq<?>, !quake.ref, !quake.ref) -> ()
// CHECK:             quake.x %[[ANC]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// clang-format on

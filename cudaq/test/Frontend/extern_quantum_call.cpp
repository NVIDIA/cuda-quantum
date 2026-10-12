/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake -I %cudaq_src_dir/unittests/nvqpp/backends/quake_backend %s | FileCheck %s
// RUN: cudaq-quake -I %cudaq_src_dir/unittests/nvqpp/backends/quake_backend %s | cudaq-opt --cable-rough-in --memtoreg | \
// RUN:   FileCheck --check-prefix=WIRE %s
// clang-format on

#include <cudaq.h>

// A quantum operation the backend implements. The declaration is the one that
// the backend ships, which the unit tests of the backend use too. The classical
// argument comes first, the order the callee results are indexed against.
#include "qm_wait.h"

__qpu__ void ramsey(double d) {
  cudaq::qubit q;
  rx(M_PI_2, q);
  __qm__wait_function(d, q);
  rx(M_PI_2, q);
  mz(q);
}

// The bridge emits an ordinary call in reference form.

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_ramsey.
// CHECK: %[[R:.*]] = quake.alloca !quake.ref
// CHECK: quake.rx
// CHECK: call @__qm__wait_function(%{{.*}}, %[[R]]) :
// CHECK-SAME: (f64, !quake.ref) -> ()
// CHECK: quake.rx
// CHECK: func.func private @__qm__wait_function(f64, !quake.ref)

// `cable-rough-in` rewrites the call into wire form. This is the acceptance
// criterion for the design.

// WIRE-LABEL: func.func @__nvqpp__mlirgen__function_ramsey.
// WIRE: %[[W0:.*]] = quake.null_wire
// WIRE: %[[W1:.*]] = quake.rx {{.*}} %[[W0]] :
// WIRE: %[[W2:.*]] = quake.apply @__qm__wait_function (%{{.*}}, %[[W1]]) :
// WIRE-SAME: (f64, !quake.wire) -> !quake.wire
// WIRE: quake.rx {{.*}} %[[W2]] :
// WIRE: func.func private @__qm__wait_function(f64, !quake.ref)

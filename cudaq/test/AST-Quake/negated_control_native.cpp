/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | \
// RUN:   cudaq-translate --convert-to=qir --preserve-gate-control-polarity | \
// RUN:   FileCheck %s

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | \
// RUN:   not cudaq-translate --convert-to=qir-base \
// RUN:     --preserve-gate-control-polarity 2>&1 | \
// RUN:   FileCheck --check-prefix=PROFILE %s

#include <cudaq.h>

struct NativeNegatedControls {
  void operator()() __qpu__ {
    cudaq::qarray<5> qreg;
    // Mixed negated and positive controls on a built-in gate.
    x<cudaq::ctrl>(!qreg[0], qreg[1], !qreg[2], qreg[3], qreg[4]);
  }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__NativeNegatedControls()
// CHECK:         @generalizedInvokeWithControlValues({{.*}}@__quantum__qis__x__ctl
// CHECK-NOT:     tail call void @__quantum__qis__x(  // unprefixed x (not x__ctl) would indicate X-expansion of the negated control

// PROFILE: `--preserve-gate-control-polarity` is only valid for full-QIR targets.

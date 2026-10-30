/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | FileCheck --check-prefix=DEFAULT %s
// RUN: cudaq-quake --quantum-opt-level=2 %s | FileCheck --check-prefix=LEVEL2 %s
// RUN: not cudaq-quake --quantum-opt-level=4 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=INVALID %s
// clang-format on

#include "cudaq/qis/qubit_qis.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
}

// clang-format off
// DEFAULT-LABEL: func.func @__nvqpp__mlirgen__function_kernel
// DEFAULT-NOT:   quake.optimizationLevel

// LEVEL2:        func.func @__nvqpp__mlirgen__function_kernel{{.*}} attributes {{.*}}quake.optimizationLevel = 2 : i32

// INVALID:       Invalid --quantum-opt-level 4

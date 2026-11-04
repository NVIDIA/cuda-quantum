/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --kernel-execution | \
// RUN:   FileCheck --check-prefix=DEFAULT %s
// RUN: cudaq-quake %s | cudaq-opt --kernel-execution=quantum-opt-level=1 | \
// RUN:   FileCheck --check-prefix=LEVEL1 %s
// RUN: cudaq-quake %s | not cudaq-opt --kernel-execution=quantum-opt-level=4 \
// RUN:   2>&1 | FileCheck --check-prefix=INVALID4 %s
// RUN: cudaq-quake %s | not cudaq-opt --kernel-execution=quantum-opt-level=-1 \
// RUN:   2>&1 | FileCheck --check-prefix=INVALID-1 %s
// clang-format on

// The quantum optimization level is not stamped on the kernel. It is passed to
// the runtime by the kernel launch.

#include "cudaq/qis/qubit_qis.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
}

// clang-format off
// DEFAULT-NOT:   quake.optimizationLevel
// DEFAULT:       %[[LEVEL:.*]] = arith.constant 2 : i32
// DEFAULT:       call @hybridLaunchKernel({{.*}}, %[[LEVEL]]) : ({{.*}}, i32) ->

// LEVEL1-NOT:    quake.optimizationLevel
// LEVEL1:        %[[LEVEL:.*]] = arith.constant 1 : i32
// LEVEL1:        call @hybridLaunchKernel({{.*}}, %[[LEVEL]]) : ({{.*}}, i32) ->

// INVALID4:      invalid quantum-opt-level 4
// INVALID-1:     invalid quantum-opt-level -1

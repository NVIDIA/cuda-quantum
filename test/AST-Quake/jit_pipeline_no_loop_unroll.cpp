/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline="builtin.module(hw-jit-prep-pipeline{no-loop-unroll=true},jit-deploy-pipeline{no-loop-unroll=true})" | FileCheck --check-prefix=PRESERVE %s
// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline="builtin.module(hw-jit-prep-pipeline,jit-deploy-pipeline)" | FileCheck --check-prefix=UNROLL %s
// clang-format on

#include "cudaq/qis/qubit_qis.h"
#include <cudaq.h>

__qpu__ int64_t loop_payload_stress() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  for (int pass = 0; pass < 2; ++pass)
    for (int i = 0; i < width; ++i)
      x(q[i]);

  int i = 0;
  while (i < width) {
    x(q[i]);
    ++i;
  }

  return cudaq::to_integer(mz(q));
}

// PRESERVE-LABEL: func.func @__nvqpp__mlirgen__function_loop_payload_stress
// PRESERVE:         cc.loop while
// PRESERVE:         cc.loop while
// PRESERVE:         cc.loop while
// PRESERVE:         quake.mz

// UNROLL-LABEL: func.func @__nvqpp__mlirgen__function_loop_payload_stress
// UNROLL-NOT:     cc.loop
// UNROLL:         quake.mz

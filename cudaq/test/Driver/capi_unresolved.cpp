/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Host code that launches a kernel needs runtime symbols outside the nvq++
// C-API, so `nvq++ -c` warns about it.

// RUN: nvq++ -c %s -o %t.o 2>&1 | FileCheck %s

#include "cudaq.h"

__qpu__ void bell() {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q);
}

int run() { return cudaq::sample(bell).size(); }

// clang-format off
// CHECK: warning: {{.*}}capi_unresolved.cpp: references CUDA-Q symbols outside the nvq++ C-API:
// CHECK: cudaq::getRegisteredExecutionManager()
// clang-format on

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Check that the analog launch entry points reject kernels that are not analog
// Hamiltonian kernels. The rejection is observed through the message the
// process prints on termination rather than by catching a typed exception,
// which is unreliable across shared library boundaries.
//
// clang-format off
// RUN: nvq++ %s -o %t && not --crash %t 2>&1 | FileCheck %s
// RUN: nvq++ -DLAUNCH_ASYNC %s -o %t && not --crash %t 2>&1 | FileCheck %s
// clang-format on

#include "cudaq/algorithms/evolve_internal.h"

int main() {
#ifdef LAUNCH_ASYNC
  auto result = cudaq::detail::launchAnalogKernelAsync("unexpected", "{}", 1);
#else
  auto result = cudaq::detail::launchAnalogKernel("unexpected", "{}", 1);
#endif
  return 0;
}

// CHECK: Unexpected type of kernel.

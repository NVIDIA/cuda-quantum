/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Exercise the RuntimeEndpoint launch path end-to-end through the user-facing
// cudaq::sample / cudaq::observe APIs. A mock endpoint is installed on the
// active platform; installing it discards the backing QPU, so the platform must
// fall back to a default compile target (and warn) for kernel compilation to
// succeed. The mock launch functions then confirm the launches are dispatched
// to the endpoint rather than a real QPU.
//
// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// clang-format on

#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/platform/quantum_platform.h"
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithm.h>

__qpu__ void bell() {
  cudaq::qubit q, r;
  h(q);
  x<cudaq::ctrl>(q, r);
}

using namespace cudaq;

sample_result mockSample(std::any &, const sample_policy &policy,
                         const CompiledModule &, KernelArgs) {
  std::printf("[sample] kernel=%s\n", policy.kernelName.c_str());
  return {};
}

observe_result mockObserve(std::any &, const observe_policy &policy,
                           const CompiledModule &, KernelArgs) {
  std::printf("[observe] kernel=%s\n", policy.kernelName.c_str());
  return observe_result(0.25, policy.spin);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  auto &platform = get_platform();

  RuntimeEndpoint ep;
  ep.impl = 0;
  ep.dispatch.set<sample_policy>(mockSample);
  ep.dispatch.set<observe_policy>(mockObserve);
  platform.setRuntimeEndpoint(std::move(ep));
  // Installing the endpoint discards the backing QPU, so the platform installs
  // a default compile target and warns about it.
  // CHECK: Overriding compile target with default

  (void)cudaq::sample(10, bell);
  // CHECK: [sample] kernel={{.*}}

  auto obs = cudaq::observe(10, bell, spin_op::z(0));
  std::printf("expectation=%.2f\n", obs.expectation());
  // CHECK: [observe] kernel={{.*}}
  // CHECK: expectation=0.25

  return 0;
}

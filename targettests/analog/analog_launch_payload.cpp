/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Check that the analog Hamiltonian launch path forwards the kernel name, the
// JSON payload and the shot count to the endpoint unchanged, and that the
// source module arrives without having been JIT compiled (its thunk is still
// null). A mock endpoint stands in for the analog QPU, so this runs on the
// default target and needs no credentials.
//
// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// clang-format on

#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/platform/quantum_platform.h"
#include <cstdio>
#include <future>
#include <string>

using namespace cudaq;

namespace {

int syncLaunches = 0;
int asyncLaunches = 0;

void report(const char *tag, std::size_t shots, const CompiledModule &module,
            const KernelArgs &args) {
  auto functionPtr = module.getFunctionPtr();
  std::string payload;
  if (auto packed = args.getPacked())
    payload.assign(reinterpret_cast<const char *>(packed->data.data()),
                   packed->data.size());
  std::printf("[%s] kernel=%s shots=%zu thunk=%s payload=%s\n", tag,
              module.getName().c_str(), shots,
              functionPtr && functionPtr->getFn() == nullptr ? "null"
                                                             : "compiled",
              payload.c_str());
}

sample_result mockSample(std::any &, const sample_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  ++syncLaunches;
  report("sample", static_cast<std::size_t>(policy.options.shots), module,
         args);
  return {};
}

async_sample_result mockSampleAsync(std::any &,
                                    const async_sample_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
  ++asyncLaunches;
  report("sample_async", static_cast<std::size_t>(policy.inner.options.shots),
         module, args);
  std::promise<sample_result> promise;
  promise.set_value({});
  return async_sample_result(detail::future(promise.get_future()));
}

} // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);

  RuntimeEndpoint endpoint;
  endpoint.impl = 0;
  endpoint.dispatch.set<sample_policy>(mockSample);
  endpoint.dispatch.set<async_sample_policy>(mockSampleAsync);
  get_platform().setRuntimeEndpoint(std::move(endpoint));
  // Installing the endpoint discards the backing QPU, so the platform installs
  // a default compile target and warns about it.
  // CHECK: Overriding compile target with default

  auto syncResult = detail::launchAnalogKernel(
      "__analog_hamiltonian_kernel__sync", R"({"sync":true})", 11);
  // CHECK: [sample] kernel=__analog_hamiltonian_kernel__sync shots=11
  // thunk=null payload={"sync":true}

  auto asyncResult = detail::launchAnalogKernelAsync(
      "__analog_hamiltonian_kernel__async", R"({"async":true})", 13);
  // CHECK: [sample_async] kernel=__analog_hamiltonian_kernel__async shots=13
  // thunk=null payload={"async":true}

  std::printf("shots sync=%zu async=%zu\n",
              static_cast<std::size_t>(syncResult.get_total_shots()),
              static_cast<std::size_t>(asyncResult.get().get_total_shots()));
  // CHECK: shots sync=0 async=0

  std::printf("launches sync=%d async=%d\n", syncLaunches, asyncLaunches);
  // CHECK: launches sync=1 async=1

  return 0;
}

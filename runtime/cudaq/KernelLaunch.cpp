/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/KernelLaunch.h"
#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include <type_traits>
#include <variant>

cudaq::InvocationContext cudaq::claimInvocationContext() {
  if (auto ctx = popInvocationContext())
    return *std::move(ctx);
  return getQuantumPlatformInternal()->defaultInvocationContext();
}

cudaq::KernelThunkResultType cudaq::launchKernel(
    const AnyModule &module, KernelArgs args, const CompileTarget &target,
    const CompileOptions &compileOptions, const LaunchOptions &launchOptions,
    RuntimeEndpoint &endpoint, CompileModuleFn compileModule) {
  if (launchOptions.policy) {
    CompiledModule compiled;
    if (const auto *source = std::get_if<SourceModule>(&module)) {
      if (compileModule) {
        CUDAQ_INFO("No compiled module found. Compiling.");
        compiled = compileModule(target, compileOptions, *source, args,
                                 /*isEntryPoint=*/true);
      } else {
        // If JIT compilation is disabled, compilation is a no-op. QPUs may
        // throw an error if they expect a JIT-compiled module.
        CUDAQ_INFO("JIT compilation is disabled. Compilation is a no-op.");
        compiled = CompiledModule{*source};
      }
    } else {
      CUDAQ_INFO("Found compiled module. Skipping compilation.");
      compiled = std::get<CompiledModule>(module);
    }

    CUDAQ_INFO("Launching kernel on QPU {}.", launchOptions.qpuId);
    std::visit(
        [&](const auto &policy) {
          using Policy = std::decay_t<decltype(policy)>;
          auto *result =
              static_cast<typename Policy::result_type *>(launchOptions.result);
          detail::with_policy_and_ctx(
              policy, *launchOptions.executionContext,
              [&] { *result = endpoint.launchKernel(policy, compiled, args); });
        },
        *launchOptions.policy);
    return {};
  }
  auto *qpu = endpoint.getQPU();
  if (qpu == nullptr)
    throw std::runtime_error(
        "This runtime endpoint does not support launching a kernel without a "
        "launch policy");
  return qpu->compileAndLaunchKernel(module, args, target, compileOptions);
}

/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteRESTQPU.h"

using namespace cudaq;
cudaq::RemoteRESTQPU::~RemoteRESTQPU() = default;

sample_result RemoteRESTQPU::launchKernel(sample_policy &policy,
                                          const AnyModule &module,
                                          KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);
  auto [kernelName, codes] = compileKernelExecutions(policy, module, args);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

async_sample_result RemoteRESTQPU::launchKernel(async_sample_policy &policy,
                                                const AnyModule &module,
                                                KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);
  auto [kernelName, codes] =
      compileKernelExecutions(policy.inner, module, args);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

KernelThunkResultType
RemoteRESTQPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  Compiler compiler(getCompileTarget(getExecutionContext()));

  std::string kernelName;
  std::optional<CompiledModule> compiled;

  if (std::holds_alternative<SourceModule>(module)) {
    const auto &src = std::get<SourceModule>(module);
    kernelName = src.getName();
    CUDAQ_INFO("launching remote rest kernel ({})", kernelName);

    auto [moduleOp, context] = Compiler::loadQuakeCodeByName(kernelName);

    // Get the Quake code, lowered according to config file.
    compiled = compiler.runPassPipeline(kernelName, moduleOp, args, true,
                                        std::move(context));
  } else {
    compiled = std::get<CompiledModule>(module);
    kernelName = compiled->getName();
    CUDAQ_INFO("launching remote rest kernel via module ({})", kernelName);
  }

  auto codes = compiler.emitKernelExecutions(*compiled);

  // Propagate metadata from the compiled artifact to the execution context.
  if (auto ctx = getExecutionContext()) {
    ctx->hasConditionalsOnMeasureResults =
        compiled->getMetadata().hasConditionalsOnMeasureResults;
  }

  completeLaunchKernel(kernelName, std::move(codes));
  return {};
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteRESTQPU, remote_rest)

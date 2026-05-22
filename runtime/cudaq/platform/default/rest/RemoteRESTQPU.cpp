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
  std::string kernelName;
  auto codes = compileKernelExecutions(policy, module, args, kernelName);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

async_sample_result RemoteRESTQPU::launchKernel(async_sample_policy &policy,
                                                const AnyModule &module,
                                                KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);
  std::string kernelName;
  auto codes = compileKernelExecutions(policy.inner, module, args, kernelName);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

KernelThunkResultType
RemoteRESTQPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  Compiler compiler(serverHelper.get(), backendConfig, targetConfig, noiseModel,
                    emulate);
  std::string kernelName;
  std::vector<cudaq::KernelExecution> codes;

  if (std::holds_alternative<SourceModule>(module)) {
    const auto &src = std::get<SourceModule>(module);
    kernelName = src.getName();
    CUDAQ_INFO("launching remote rest kernel ({})", kernelName);

    auto executionContext = cudaq::getExecutionContext();

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    auto [moduleOp, context] = Compiler::loadQuakeCodeByName(kernelName);

    // Get the Quake code, lowered according to config file.
    codes =
        compiler.lowerQuakeCode(executionContext, kernelName, moduleOp, args);
  } else {
    const auto &compiled = std::get<CompiledModule>(module);
    kernelName = compiled.getName();
    CUDAQ_INFO("launching remote rest kernel via module ({})", kernelName);
    codes = compiler.emitKernelExecutions(compiled);
  }

  completeLaunchKernel(kernelName, std::move(codes));
  return {};
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteRESTQPU, remote_rest)

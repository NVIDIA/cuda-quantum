/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteRESTQPU.h"
#include "common/CompiledModule.h"
#include "common/KernelExecution.h"
#include "cudaq_internal/compiler/Compiler.h"

static std::vector<cudaq::KernelExecution>
getKernelExecutions(const cudaq::CompiledModule &module,
                    bool requireTargetCode = true) {
  if (module.getMlirArtifacts().empty())
    CUDAQ_ERROR("QPU does not support launching a "
                "CompiledModule without MLIR artifacts.");

  return cudaq_internal::compiler::CompiledModuleHelper::toKernelExecutions(
      module, requireTargetCode);
}

using namespace cudaq;
cudaq::RemoteRESTQPU::~RemoteRESTQPU() = default;

sample_result RemoteRESTQPU::launchKernel(const sample_policy &policy,
                                          const CompiledModule &module,
                                          KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_sample_result
RemoteRESTQPU::launchKernel(const async_sample_policy &policy,
                            const CompiledModule &module, KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

observe_result RemoteRESTQPU::launchKernel(const observe_policy &policy,
                                           const CompiledModule &module,
                                           KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

run_result RemoteRESTQPU::launchKernel(const run_policy &policy,
                                       const CompiledModule &module,
                                       KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_run_result RemoteRESTQPU::launchKernel(const async_run_policy &policy,
                                             const CompiledModule &module,
                                             KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_observe_result
RemoteRESTQPU::launchKernel(const async_observe_policy &policy,
                            const CompiledModule &module, KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto codes = getKernelExecutions(module);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

KernelThunkResultType
RemoteRESTQPU::compileAndLaunchKernel(const AnyModule &module, KernelArgs args,
                                      const CompileTarget &target,
                                      const CompileOptions &options) {
  CompiledModule compiled;
  if (std::holds_alternative<SourceModule>(module)) {
    const auto &source = std::get<SourceModule>(module);
    CUDAQ_INFO("no compiled kernel found for {}, compiling now",
               source.getName());
    compiled =
        cudaq_internal::compiler::compileModule(target, options, source, args);
  } else {
    compiled = std::get<CompiledModule>(module);
  }
  CUDAQ_INFO("launching remote rest kernel ({})", compiled.getName());

  auto codes = getKernelExecutions(compiled, /*requireTargetCode=*/false);
  completeLaunchKernel(compiled.getName(), std::move(codes));
  return {};
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteRESTQPU, remote_rest)

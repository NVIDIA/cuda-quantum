/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteRESTQPU.h"
#include "cudaq_internal/compiler/Compiler.h"

using namespace cudaq;
cudaq::RemoteRESTQPU::~RemoteRESTQPU() = default;

static std::vector<cudaq::KernelExecution>
runCodegen(cudaq_internal::compiler::Compiler &compiler,
           const CompiledModule &module, KernelArgs args) {
  // TODO: This should be moved into compiler::compileModule, but this would add
  // a dependency on the compiler in the C++ launch path.
  auto compiled = module;
  cudaq_internal::compiler::CompiledModuleHelper::ensureMlirArtifactsExist(
      compiled, compiler, args);

  return compiler.emitKernelExecutions(compiled);
}

sample_result RemoteRESTQPU::launchKernel(const sample_policy &policy,
                                          const CompiledModule &module,
                                          KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy));
  auto codes = runCodegen(compiler, module, args);

  if (compiler.hasWarnedNamedMeasurements())
    policy.warnedNamedMeasurements = true;
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_sample_result
RemoteRESTQPU::launchKernel(const async_sample_policy &policy,
                            const CompiledModule &module, KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy.inner));
  auto codes = runCodegen(compiler, module, args);

  if (compiler.hasWarnedNamedMeasurements())
    policy.inner.warnedNamedMeasurements = true;
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

observe_result RemoteRESTQPU::launchKernel(const observe_policy &policy,
                                           const CompiledModule &module,
                                           KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy));
  auto codes = runCodegen(compiler, module, args);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_observe_result RemoteRESTQPU::launchKernel(async_observe_policy &policy,
                                                 const CompiledModule &module,
                                                 KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy.inner));
  auto codes = runCodegen(compiler, module, args);
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

KernelThunkResultType
RemoteRESTQPU::unifiedLaunchModule(const CompiledModule &module,
                                   KernelArgs args) {
  CUDAQ_INFO("launching remote rest kernel ({})", module.getName());

  cudaq_internal::compiler::Compiler compiler(
      getCompileTarget(other_policies{}, getExecutionContext()));
  auto codes = runCodegen(compiler, module, args);

  completeLaunchKernel(module.getName(), std::move(codes));
  return {};
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteRESTQPU, remote_rest)

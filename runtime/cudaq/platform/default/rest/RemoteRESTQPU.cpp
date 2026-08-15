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
#include "cudaq/algorithms/policy_cpos.h"

static std::vector<cudaq::KernelExecution>
runCodegen(const cudaq::CompiledModule &module, cudaq::CompileTarget target) {
  auto mlirArtifacts = module.getMlirArtifacts();
  if (mlirArtifacts.empty())
    CUDAQ_ERROR("QPU does not support launching a "
                "CompiledModule without MLIR artifacts.");

  cudaq_internal::compiler::Compiler compiler(std::move(target), {});

  // Pre-compiled modules (e.g. Python kernels) have been through the generic
  // aot-prep-pipeline but NOT through the target's jit-mid-level-pipeline.
  // Apply the target's full pipeline to each MLIR artifact now so that
  // target-specific passes (e.g. add-wireset + assign-wire-indices for the
  // quake_fake backend) are applied before the IR is serialized and sent to
  // the server.
  std::vector<cudaq::KernelExecution> allCodes;
  for (const auto &[name, artifact] : mlirArtifacts) {
    auto compiled =
        compiler.runPassPipeline(name, artifact.getOpaqueModulePtr(), {},
                                 /*isEntryPoint=*/true, artifact.getContext());
    auto codes = compiler.emitKernelExecutions(compiled);
    allCodes.insert(allCodes.end(), codes.begin(), codes.end());
  }
  return allCodes;
}

using namespace cudaq;
cudaq::RemoteRESTQPU::~RemoteRESTQPU() = default;

sample_result RemoteRESTQPU::launchKernel(const sample_policy &policy,
                                          const CompiledModule &module,
                                          KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto target = getCompileTarget(policy);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_sample_result
RemoteRESTQPU::launchKernel(const async_sample_policy &policy,
                            const CompiledModule &module, KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto target = getCompileTarget(policy.inner);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

observe_result RemoteRESTQPU::launchKernel(const observe_policy &policy,
                                           const CompiledModule &module,
                                           KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto target = getCompileTarget(policy);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

run_result RemoteRESTQPU::launchKernel(const run_policy &policy,
                                       const CompiledModule &module,
                                       KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel {}", policy.name);

  auto target = getCompileTarget(policy);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_run_result RemoteRESTQPU::launchKernel(const async_run_policy &policy,
                                             const CompiledModule &module,
                                             KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto target = getCompileTarget(policy.inner);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

async_observe_result
RemoteRESTQPU::launchKernel(const async_observe_policy &policy,
                            const CompiledModule &module, KernelArgs args) {
  CUDAQ_INFO("RemoteRESTQPU::launchKernel async {}", policy.inner.name);

  auto target = getCompileTarget(policy.inner);
  auto codes = runCodegen(module, std::move(target));
  return completeLaunchKernel(policy, module.getName(), std::move(codes));
}

KernelThunkResultType
RemoteRESTQPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  auto *ctx = getExecutionContext();
  CompiledModule compiled;
  auto target = getCompileTarget(other_policies{}, ctx);
  CompileOptions options = cudaq::get_compile_options(other_policies{});
  cudaq_internal::compiler::Compiler compiler(std::move(target),
                                              std::move(options));

  if (std::holds_alternative<SourceModule>(module)) {
    const auto &source = std::get<SourceModule>(module);
    CUDAQ_INFO("no compiled kernel found for {}, compiling now",
               source.getName());
    auto mlirArt =
        cudaq_internal::compiler::CompiledModuleHelper::loadMlirArtifact(
            source);
    compiled =
        compiler.runPassPipeline(source.getName(), mlirArt.getOpaqueModulePtr(),
                                 args, true, mlirArt.getContext());
  } else {
    // Pre-compiled module: apply the target's pipeline (same reason as in
    // runCodegen — the aot-prep-pipeline doesn't run the target's
    // jit-mid-level-pipeline).
    const auto &precompiled = std::get<CompiledModule>(module);
    if (precompiled.getMlirArtifacts().empty())
      CUDAQ_ERROR("QPU does not support launching a "
                  "CompiledModule without MLIR artifacts.");
    const auto &[artName, artifact] =
        precompiled.getMlirArtifacts().front();
    compiled = compiler.runPassPipeline(
        artName, artifact.getOpaqueModulePtr(), args,
        /*isEntryPoint=*/true, artifact.getContext());
  }
  CUDAQ_INFO("launching remote rest kernel ({})", compiled.getName());

  if (compiled.getMlirArtifacts().empty())
    CUDAQ_ERROR("QPU does not support launching a "
                "CompiledModule without MLIR artifacts.");

  auto codes = compiler.emitKernelExecutions(compiled);

  completeLaunchKernel(compiled.getName(), std::move(codes));
  return {};
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteRESTQPU, remote_rest)

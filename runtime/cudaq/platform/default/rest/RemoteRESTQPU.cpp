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
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/algorithms/policy_cpos.h"
#include "mlir/IR/BuiltinOps.h"

/// Return true if \p modulePtr already contains a quake.wire_set — indicating
/// that the target's jit-mid-level-pipeline has already been applied (e.g. by
/// the nvq++ AOT compilation step for C++ kernels).
static bool moduleContainsWireSet(const void *modulePtr) {
  auto mod = mlir::ModuleOp::getFromOpaquePointer(modulePtr);
  return mod
      ->walk<mlir::WalkOrder::PreOrder>(
          [](cudaq::quake::WireSetOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

static std::vector<cudaq::KernelExecution>
runCodegen(const cudaq::CompiledModule &module, cudaq::CompileTarget target) {
  auto mlirArtifacts = module.getMlirArtifacts();
  if (mlirArtifacts.empty())
    CUDAQ_ERROR("QPU does not support launching a "
                "CompiledModule without MLIR artifacts.");

  // Determine whether this target requires the jit-mid-level-pipeline to have
  // been applied (indicated by a non-empty midLevelPipeline config).  Most
  // targets only need the generic aot-prep-pipeline; quake_fake-style targets
  // additionally require add-wireset / assign-wire-indices.
  const bool targetNeedsMidLevel =
      !target.pipelineConfig.midLevelPipeline.empty();

  cudaq_internal::compiler::Compiler compiler(std::move(target), {});

  // If the target has a mid-level pipeline and the module has not yet been
  // processed by it (no quake.wire_set → pre-compiled Python kernel), apply
  // the full target pipeline now.  C++ kernels compiled by nvq++ already carry
  // a quake.wire_set from their AOT compilation and must not be processed
  // again.
  std::vector<cudaq::KernelExecution> allCodes;
  for (const auto &[name, artifact] : mlirArtifacts) {
    if (targetNeedsMidLevel &&
        !moduleContainsWireSet(artifact.getOpaqueModulePtr())) {
      auto compiled = compiler.runPassPipeline(
          name, artifact.getOpaqueModulePtr(), {},
          /*isEntryPoint=*/true, artifact.getContext());
      auto codes = compiler.emitKernelExecutions(compiled);
      allCodes.insert(allCodes.end(), codes.begin(), codes.end());
    } else {
      // Already processed or target does not require mid-level pipeline.
      auto codes = compiler.emitKernelExecutions(module);
      allCodes.insert(allCodes.end(), codes.begin(), codes.end());
      break; // emitKernelExecutions processes all artifacts at once
    }
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
    compiled = std::get<CompiledModule>(module);
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

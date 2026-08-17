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
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/algorithms/policy_cpos.h"
#include "mlir/IR/BuiltinOps.h"

/// Return true if \p modulePtr is a Python-compiled kernel.  Python kernels
/// carry the `quake.python_uniqued` attribute (set by the Python bridge) and
/// have only been through the generic `aot-prep-pipeline`.  C++ kernels
/// compiled by nvq++ do not carry this attribute and have already been through
/// the full target-specific pipeline (including any jit-mid-level-pipeline).
static bool isPythonKernel(const void *modulePtr) {
  auto mod = mlir::ModuleOp::getFromOpaquePointer(modulePtr);
  return mod->hasAttr(cudaq::runtime::pythonUniqueAttrName);
}

static std::vector<cudaq::KernelExecution>
runCodegen(const cudaq::CompiledModule &module, cudaq::CompileTarget target) {
  auto mlirArtifacts = module.getMlirArtifacts();
  if (mlirArtifacts.empty())
    CUDAQ_ERROR("QPU does not support launching a "
                "CompiledModule without MLIR artifacts.");

  // For Python kernels the aot-prep-pipeline has been applied but NOT the
  // target's jit-mid-level-pipeline.  runPassPipeline is needed in two cases:
  //   1. The target sends raw Quake MLIR (codegenTranslation == "nop", e.g.
  //      quake_fake) and requires wireset/borrow-wire ops from the mid-level.
  //   2. The target runs locally via emulation and needs a JIT artifact.  The
  //      aot-prep-pipeline never creates a JIT (emitJit defaults to false), so
  //      the cached CompiledModule has no JIT; we must produce one here.
  //
  // C++ kernels compiled by nvq++ have already been through the full pipeline
  // and must NOT be reprocessed — doing so would double-apply passes.
  // Determine what the module needs that the aot-prep-pipeline didn't provide:
  //   • needsWireset: target sends raw Quake MLIR (codegenTranslation=="nop")
  //     and the server requires add-wireset / assign-wire-indices ops.
  //   • needsJit: target runs locally via emulation and a JIT artifact is
  //     required for cudaq::sample / run.  The aot-prep-pipeline produces
  //     compiled modules with emitJit=false, so there is no JIT yet.
  //
  // If the module already has a JIT (e.g. compileModule was called earlier in
  // the same launch with emitJit=true from the policy options), we must NOT
  // call runPassPipeline again — that would double-apply mapping/decomposition
  // passes and produce wrong results.
  //
  // C++ kernels compiled by nvq++ have been through the full pipeline already
  // and must not be reprocessed.
  const bool isPython =
      !mlirArtifacts.empty() &&
      isPythonKernel(mlirArtifacts.front().second.getOpaqueModulePtr());
  const bool needsWireset = target.pipelineConfig.codegenTranslation == "nop" &&
                            !target.pipelineConfig.midLevelPipeline.empty();
  const bool needsJit = target.emulate && !module.getJit().has_value();

  if (isPython && (needsWireset || needsJit)) {
    cudaq::CompileOptions opts;
    opts.emitJit = needsJit;
    cudaq_internal::compiler::Compiler compiler(std::move(target), opts);
    std::vector<cudaq::KernelExecution> allCodes;
    for (const auto &[name, artifact] : mlirArtifacts) {
      auto compiled = compiler.runPassPipeline(
          name, artifact.getOpaqueModulePtr(), {},
          /*isEntryPoint=*/true, artifact.getContext());
      auto codes = compiler.emitKernelExecutions(compiled);
      allCodes.insert(allCodes.end(), codes.begin(), codes.end());
    }
    return allCodes;
  }

  cudaq_internal::compiler::Compiler compiler(std::move(target), {});
  return compiler.emitKernelExecutions(module);
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

/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "algorithms/observe/policy.h"
#include "algorithms/policies.h"
#include "algorithms/sample/policy.h"
#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "common/Timing.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstring>
#include <stdexcept>

using namespace cudaq_internal::compiler;
using namespace cudaq;

cudaq::KernelThunkResultType
cudaq::QPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  if (std::holds_alternative<SourceModule>(module))
    throw std::runtime_error(
        "This QPU does not support launching uncompiled SourceModule kernels; "
        "subclasses must override unifiedLaunchModule.");

  const auto &compiled = std::get<CompiledModule>(module);
  return runJITCompiledModule(compiled, args);
}

sample_result cudaq::QPU::launchKernel(const sample_policy &policy,
                                       const AnyModule &module,
                                       KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the sample_policy.");
}

async_sample_result cudaq::QPU::launchKernel(const async_sample_policy &policy,
                                             const AnyModule &module,
                                             KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the async_sample_policy.");
}

observe_result cudaq::QPU::launchKernel(const observe_policy &policy,
                                        const AnyModule &module,
                                        KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the observe_policy.");
}

async_observe_result cudaq::QPU::launchKernel(async_observe_policy &policy,
                                              const AnyModule &module,
                                              KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the async_observe_policy.");
}

cudaq::KernelThunkResultType
cudaq::QPU::runJITCompiledModule(const CompiledModule &compiled,
                                 KernelArgs args) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::runJITCompiledModule",
                         compiled.getName());

  // Propagate metadata from the compiled artifact to the execution context.
  if (auto ctx = getExecutionContext()) {
    ctx->hasConditionalsOnMeasureResults =
        compiled.getMetadata().hasConditionalsOnMeasureResults;

    if (ctx->name == "resource-count" && compiled.getResources()) {
      nvqir::resource_counter::prepopulate(*compiled.getResources());
    }
  }

  auto rawArgs = args.getTypeErased().value_or(std::span<void *const>{});
  auto funcPtr = compiled.getJit()->getFn();
  const auto &resultInfo = compiled.getResultInfo();
  if (!compiled.isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    auto argsCreator = compiled.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    // If the kernel has a result, copy it from the packed buffer into
    // rawArgs.back() (where the caller expects to find it).
    if (resultInfo.hasResult()) {
      auto offset = compiled.getReturnOffset().value();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.getBufferSize());
    }
    std::free(buff);
    return {nullptr, 0};
  }
  if (resultInfo.hasResult()) {
    // Fully specialized with result: rawArgs.back() is the pre-allocated
    // result buffer; pass it directly to the thunk.
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  // Fully specialized, no result.
  funcPtr();
  return {nullptr, 0};
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::QPU::getCompileTarget(const sample_policy &) {
  throw std::runtime_error(
      "no CompileTarget defined for sample_policy on this QPU");
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::QPU::getCompileTarget(const observe_policy &) {
  throw std::runtime_error(
      "no CompileTarget defined for observe_policy on this QPU");
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::QPU::getCompileTarget(const other_policies &olicy, ExecutionContext *) {
  throw std::runtime_error(
      "no CompileTarget defined for other_policies on this QPU");
}

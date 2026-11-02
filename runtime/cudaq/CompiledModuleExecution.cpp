/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "common/Timing.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "cudaq/runtime/logger/logger.h"
#include <cstring>

extern "C" void __quantum__rt__clear_result_maps();

namespace {

/// RAII marker for the window in which a JIT/AOT-compiled kernel frame is
/// executing. While alive it sets `ExecutionContext::inKernelLaunch` on the
/// active context, so the simulator defers (rather than throws) exceptions that
/// would otherwise have to unwind through the kernel frame.
struct InKernelLaunchScope {
  InKernelLaunchScope() {
    if (auto *ctx = cudaq::getExecutionContext())
      ctx->inKernelLaunch = true;
  }
  ~InKernelLaunchScope() {
    if (auto *ctx = cudaq::getExecutionContext())
      ctx->inKernelLaunch = false;
  }
  InKernelLaunchScope(const InKernelLaunchScope &) = delete;
  InKernelLaunchScope &operator=(const InKernelLaunchScope &) = delete;
};

} // namespace

using namespace cudaq;

KernelThunkResultType cudaq::executeCompiledModule(const CompiledModule &module,
                                                   KernelArgs args) {
  auto rawFn = module.getFunctionPtr();
  if (rawFn)
    return executeFunctionPtrBinary(*rawFn, args);
  return executeJitBinary(module, args);
}

KernelThunkResultType cudaq::executeJitBinary(const CompiledModule &module,
                                              KernelArgs args) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "executeJitBinary",
                         module.getName());

  // Propagate metadata from the compiled artifact to the execution context.
  if (auto ctx = getExecutionContext()) {
    ctx->hasConditionalsOnMeasureResults =
        module.getMetadata().hasConditionalsOnMeasureResults;

    if (ctx->name == "resource-count" && module.getResources()) {
      nvqir::resource_counter::prepopulate(*module.getResources());
    }
  }

  auto rawArgs = args.getTypeErased().value_or(std::span<void *const>{});
  auto funcPtr = module.getJit()->getFn();
  const auto &resultInfo = module.getResultInfo();
  // Mark the kernel frame so the simulator defers (rather than throws)
  // exceptions while the JIT'd kernel runs; rethrowDeferredKernelException()
  // below surfaces any such error from this C++ frame.
  InKernelLaunchScope kernelFrame;
  if (!module.isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    auto argsCreator = module.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    // If the kernel has a result, copy it from the packed buffer into
    // rawArgs.back() (where the caller expects to find it).
    if (resultInfo.hasResult()) {
      auto offset = module.getReturnOffset().value();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.getBufferSize());
    }
    std::free(buff);
    rethrowDeferredKernelException();
    return {nullptr, 0};
  }
  if (resultInfo.hasResult()) {
    // Fully specialized with result: rawArgs.back() is the pre-allocated
    // result buffer; pass it directly to the thunk.
    void *buff = const_cast<void *>(rawArgs.back());
    auto result = reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(
        funcPtr)(buff, /*client_server=*/false);
    rethrowDeferredKernelException();
    return result;
  }
  // Fully specialized, no result.
  funcPtr();
  rethrowDeferredKernelException();
  return {nullptr, 0};
}

KernelThunkResultType cudaq::executeFunctionPtrBinary(
    const FatQuakeModule::FunctionPtrArtifact &artifact, KernelArgs args) {
  auto packed = args.getPacked();
  void *argData = packed ? packed->data.data() : nullptr;
  // Mark the kernel frame so the simulator defers (rather than throws)
  // exceptions while the AOT kernel runs; rethrowDeferredKernelException()
  // below surfaces any such error from this C++ frame.
  InKernelLaunchScope kernelFrame;
  // A function-pointer artifact (an nvq++ AOT thunk or a C++ `kernel_builder`
  // thunk) carries no result-map cleanup. This launch clears NVQIR's result
  // maps before and after the kernel runs, as the JIT does for its entry points
  // (`insertResultMapCleanupOperations`), even when the kernel throws. Nested
  // kernel calls run inside the launch and keep their measurement handles.
  __quantum__rt__clear_result_maps();
  auto result = detail::try_finally(
      [&] { return artifact.getFn()(argData, /*isRemote=*/false); },
      [] { __quantum__rt__clear_result_maps(); });
  rethrowDeferredKernelException();
  return result;
}

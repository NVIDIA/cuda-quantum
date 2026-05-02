/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "common/CompiledModule.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstring>

using namespace cudaq_internal::compiler;

CUDAQ_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

/// Execute a JIT-compiled kernel with provided arguments.
///
/// Handles argument marshaling via `argsCreator` (if not fully specialized) and
/// result buffer allocation.
cudaq::KernelThunkResultType
launchCompiledModule(const cudaq::CompiledModule &compiled,
                     const std::vector<void *> &rawArgs) {
  auto funcPtr = compiled.getJit()->getFn();
  const auto &resultInfo = compiled.getResultInfo();
  if (!compiled.isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    auto argsCreator = compiled.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(funcPtr)(
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
    return reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(
        funcPtr)(buff, /*client_server=*/false);
  }
  // Fully specialized, no result.
  funcPtr();
  return {nullptr, 0};
}

cudaq::KernelThunkResultType
cudaq::QPU::launchModule(const CompiledModule &module,
                         const std::vector<void *> &rawArgs) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `launchModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule",
                         module.getName());
  return launchCompiledModule(module, rawArgs);
}

cudaq::CompiledModule
cudaq::QPU::compileModule(const std::string &name, const void *modulePtr,
                          const std::vector<void *> &rawArgs,
                          bool isEntryPoint) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `compileModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::compileModule", name);
  mlir::ModuleOp module = mlir::ModuleOp::getFromOpaquePointer(modulePtr);
  return launcher->compileModule(name, module, rawArgs, isEntryPoint);
}

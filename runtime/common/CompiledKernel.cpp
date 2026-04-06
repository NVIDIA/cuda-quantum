/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"
#include <cstring>

using namespace cudaq_internal::compiler;

cudaq::CompiledKernel::CompiledKernel(
    JitEngine engine, std::string kernelName, void (*entryPoint)(),
    ArgsCreatorFunc argsCreator, ReturnOffsetFunc returnOffset,
    ResultSizeFunc resultSize, HybridLaunchFunc hybridLauncher, bool hasResult)
    : engine(engine), name(std::move(kernelName)), entryPoint(entryPoint),
      argsCreator(argsCreator), returnOffset(returnOffset),
      resultSize(resultSize), hybridLauncher(hybridLauncher),
      hasResult(hasResult) {}

cudaq::KernelThunkResultType
cudaq::CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto kernelThunk = reinterpret_cast<KernelThunkType>(getEntryPoint());
  // If there's a result, we must go through the hybrid launcher
  if (hasResult) {
    // TODO: Is this performance hack really buying us anything?
    if (!argsCreator) {
      void *buff = const_cast<void *>(rawArgs.back());
      return kernelThunk(buff, /*client_server=*/false);
    }

    void *buff = nullptr;
    auto buffSize =
        argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    auto resSize = resultSize();
    auto offset = returnOffset();
    hybridLauncher(name.c_str(), kernelThunk, buff, buffSize, offset, rawArgs);
    memcpy(rawArgs.back(), (char *)buff + offset, resSize);
    std::free(buff);
    return {nullptr, 0};
  }

  // No return value, build the argument message buffer and launch the thunk
  if (argsCreator) {
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    kernelThunk(buff, false);
    std::free(buff);
    return {nullptr, 0};
  }

  // No return value or arguments, just launch the entry function directly
  getEntryPoint()();
  return {nullptr, 0};
}

void (*cudaq::CompiledKernel::getEntryPoint() const)() { return entryPoint; }

JitEngine cudaq::CompiledKernel::getEngine() const { return engine; }

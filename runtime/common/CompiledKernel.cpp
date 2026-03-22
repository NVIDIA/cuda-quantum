/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"
#include <memory>
#include <stdexcept>

cudaq::CompiledKernel::CompiledKernel(
    JitEngine engine, std::string kernelName, void (*entryPoint)(),
    int64_t (*argsCreator)(const void *, void **), ResultInfo resultInfo)
    : engine(engine), name(std::move(kernelName)), entryPoint(entryPoint),
      argsCreator(argsCreator), resultInfo(std::move(resultInfo)) {}

cudaq::KernelThunkResultType
cudaq::CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto funcPtr = getEntryPoint();
  if (resultInfo.hasResult()) {
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  if (argsCreator) {
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    std::free(buff);
    return {nullptr, 0};
  }

  funcPtr();
  return {nullptr, 0};
}

cudaq::KernelThunkResultType cudaq::CompiledKernel::execute() const {
  if (argsCreator)
    throw std::runtime_error(
        "Kernel has unspecialized parameters; call execute(rawArgs) instead.");
  if (!resultInfo.hasResult()) {
    entryPoint();
    return {nullptr, 0};
  }
  // Allocate a result buffer on-the-fly.
  auto buf = std::make_unique<char[]>(resultInfo.bufferSize);
  std::vector<void *> rawArgs = {buf.get()};
  execute(rawArgs);
  return {buf.release(), resultInfo.bufferSize};
}

void (*cudaq::CompiledKernel::getEntryPoint() const)() { return entryPoint; }

cudaq::JitEngine cudaq::CompiledKernel::getEngine() const { return engine; }

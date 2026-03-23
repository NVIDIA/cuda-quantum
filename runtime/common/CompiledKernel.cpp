/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"

cudaq::CompiledKernel::CompiledKernel(
    JitEngine engine, std::string kernelName, void (*entryPoint)(),
    int64_t (*argsCreator)(const void *, void **), bool hasResult)
    : engine(engine), name(std::move(kernelName)), entryPoint(entryPoint),
      argsCreator(argsCreator), hasResult(hasResult) {}

cudaq::KernelThunkResultType
cudaq::CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto funcPtr = getEntryPoint();
  if (hasResult) {
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

void (*cudaq::CompiledKernel::getEntryPoint() const)() { return entryPoint; }

cudaq::JitEngine cudaq::CompiledKernel::getEngine() const { return engine; }

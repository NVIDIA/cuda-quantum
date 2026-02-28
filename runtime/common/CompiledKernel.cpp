/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"

namespace cudaq {

CompiledKernel::CompiledKernel(OpaquePtr<JitEngine> engine,
                               std::string kernelName, void (*entryPoint)(),
                               bool hasResult)
    : engine(std::move(engine)), name(std::move(kernelName)),
      entryPoint(entryPoint), hasResult(hasResult) {}

KernelThunkResultType
CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto funcPtr = getEntryPoint();
  if (hasResult) {
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  } else {
    reinterpret_cast<void (*)()>(funcPtr)();
    return {nullptr, 0};
  }
}

void (*CompiledKernel::getEntryPoint() const)() { return entryPoint; }

const JitEngine &CompiledKernel::getEngine() const { return *engine; }

} // namespace cudaq

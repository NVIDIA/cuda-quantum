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

using namespace cudaq_internal::compiler;

cudaq::CompiledKernel::CompiledKernel(std::string kernelName,
                                      ResultInfo resultInfo)
    : name(std::move(kernelName)), resultInfo(std::move(resultInfo)) {}

const cudaq::CompiledKernel::JitRepr &cudaq::CompiledKernel::getJit() const {
  if (!jitRepr)
    throw std::runtime_error("CompiledKernel has no JIT representation.");
  return *jitRepr;
}

const cudaq::CompiledKernel::MlirRepr &cudaq::CompiledKernel::getMlir() const {
  if (!mlirRepr)
    throw std::runtime_error("CompiledKernel has no MLIR representation.");
  return *mlirRepr;
}

cudaq::KernelThunkResultType
cudaq::CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto funcPtr = getEntryPoint();
  if (resultInfo.hasResult()) {
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  if (!isFullySpecialized()) {
    void *buff = nullptr;
    jitRepr->argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    std::free(buff);
    return {nullptr, 0};
  }

  funcPtr();
  return {nullptr, 0};
}

cudaq::KernelThunkResultType cudaq::CompiledKernel::execute() const {
  if (!isFullySpecialized())
    throw std::runtime_error(
        "Kernel has unspecialized parameters; call execute(rawArgs) instead.");
  if (!resultInfo.hasResult()) {
    getEntryPoint()();
    return {nullptr, 0};
  }
  // Allocate a result buffer on-the-fly.
  auto buf = std::make_unique<char[]>(resultInfo.bufferSize);
  std::vector<void *> rawArgs = {buf.get()};
  execute(rawArgs);
  return {buf.release(), resultInfo.bufferSize};
}

void (*cudaq::CompiledKernel::getEntryPoint() const)() {
  return getJit().entryPoint;
}

JitEngine cudaq::CompiledKernel::getEngine() const { return getJit().engine; }

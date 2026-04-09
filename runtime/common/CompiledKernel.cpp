/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include <cstring>
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
  if (!isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    void *buff = nullptr;
    jitRepr->argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    // If the kernel has a result, copy it from the packed buffer into
    // rawArgs.back() (where the caller expects to find it).
    if (resultInfo.hasResult()) {
      auto offset = jitRepr->returnOffset();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.bufferSize);
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
  getEntryPoint()();
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

cudaq::JitEngine cudaq::CompiledKernel::getEngine() const {
  return getJit().engine;
}

void cudaq::CompiledKernel::attachJit(JitEngine engine,
                                      bool isFullySpecialized) {
  bool hasResult = resultInfo.hasResult();
  std::string fullName = cudaq::runtime::cudaqGenPrefixName + name;
  std::string entryName =
      (hasResult || !isFullySpecialized) ? name + ".thunk" : fullName;
  void (*entryPoint)() = engine.lookupRawNameOrFail(entryName);
  int64_t (*argsCreator)(const void *, void **) = nullptr;
  int64_t (*returnOffset)() = nullptr;
  if (!isFullySpecialized) {
    argsCreator = reinterpret_cast<int64_t (*)(const void *, void **)>(
        engine.lookupRawNameOrFail(name + ".argsCreator"));
    if (hasResult)
      returnOffset = reinterpret_cast<int64_t (*)()>(
          engine.lookupRawNameOrFail(name + ".returnOffset"));
  }
  jitRepr = cudaq::CompiledKernel::JitRepr{std::move(engine), entryPoint,
                                           argsCreator, returnOffset};
}

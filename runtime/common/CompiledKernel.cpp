/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledKernel.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include <memory>
#include <stdexcept>

using namespace cudaq_internal::compiler;

cudaq::CompiledKernel::CompiledKernel(std::string kernelName,
                                      ResultInfo resultInfo)
    : name(std::move(kernelName)), resultInfo(std::move(resultInfo)) {}

const cudaq::JitArtifact &cudaq::CompiledKernel::getJit() const {
  for (auto &[key, artifact] : artifacts)
    if (auto *jit = std::get_if<JitArtifact>(&artifact))
      return *jit;
  throw std::runtime_error("CompiledKernel has no JIT artifact.");
}

const cudaq::MlirArtifact &cudaq::CompiledKernel::getMlir() const {
  for (auto &[key, artifact] : artifacts)
    if (auto *mlir = std::get_if<MlirArtifact>(&artifact))
      return *mlir;
  throw std::runtime_error("CompiledKernel has no MLIR artifact.");
}

bool cudaq::CompiledKernel::hasJit() const {
  for (auto &[key, artifact] : artifacts)
    if (std::holds_alternative<JitArtifact>(artifact))
      return true;
  return false;
}

bool cudaq::CompiledKernel::hasMlir() const {
  for (auto &[key, artifact] : artifacts)
    if (std::holds_alternative<MlirArtifact>(artifact))
      return true;
  return false;
}

bool cudaq::CompiledKernel::isFullySpecialized() const {
  for (auto &[key, artifact] : artifacts)
    if (auto *jit = std::get_if<JitArtifact>(&artifact))
      return jit->argsCreator == nullptr;
  return true; // No JIT artifact → fully specialized.
}

void cudaq::CompiledKernel::addArtifact(std::string name,
                                        CompiledArtifact artifact) {
  if (artifacts.contains(name))
    throw std::runtime_error("Artifact with name " + name + " already exists");
  artifacts.emplace(std::move(name), std::move(artifact));
}

cudaq::KernelThunkResultType
cudaq::CompiledKernel::execute(const std::vector<void *> &rawArgs) const {
  auto &jit = getJit();
  auto funcPtr = jit.entryPoint;
  if (resultInfo.hasResult()) {
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  if (!isFullySpecialized()) {
    void *buff = nullptr;
    jit.argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
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
  if (!isFullySpecialized)
    argsCreator = reinterpret_cast<int64_t (*)(const void *, void **)>(
        engine.lookupRawNameOrFail(name + ".argsCreator"));

  addArtifact(name, JitArtifact{std::move(engine), entryPoint, argsCreator,
                                std::nullopt});
}

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledModule.h"
#include <memory>
#include <stdexcept>

cudaq::CompiledModule::CompiledModule(std::string kernelName)
    : name(std::move(kernelName)) {}

const cudaq::CompiledModule::JitArtifact &
cudaq::CompiledModule::getJit() const {
  for (auto &[key, artifact] : artifacts)
    if (auto *jit = std::get_if<JitArtifact>(&artifact))
      return *jit;
  throw std::runtime_error("CompiledModule has no JIT artifact.");
}

const cudaq::CompiledModule::MlirArtifact &
cudaq::CompiledModule::getMlir() const {
  for (auto &[key, artifact] : artifacts)
    if (auto *mlir = std::get_if<MlirArtifact>(&artifact))
      return *mlir;
  throw std::runtime_error("CompiledModule has no MLIR artifact.");
}

bool cudaq::CompiledModule::hasJit() const {
  for (auto &[key, artifact] : artifacts)
    if (std::holds_alternative<JitArtifact>(artifact))
      return true;
  return false;
}

bool cudaq::CompiledModule::hasMlir() const {
  for (auto &[key, artifact] : artifacts)
    if (std::holds_alternative<MlirArtifact>(artifact))
      return true;
  return false;
}

bool cudaq::CompiledModule::isFullySpecialized() const {
  if (!hasJit())
    return true; // No JIT artifact → fully specialized.
  return getJit().argsCreator == nullptr;
}

void cudaq::CompiledModule::addArtifact(std::string name,
                                        CompiledArtifact artifact) {
  if (artifacts.contains(name))
    throw std::runtime_error("Artifact with name " + name + " already exists");
  artifacts.emplace(std::move(name), std::move(artifact));
}

cudaq::KernelThunkResultType
cudaq::CompiledModule::execute(const std::vector<void *> &rawArgs) const {
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

cudaq::KernelThunkResultType cudaq::CompiledModule::execute() const {
  if (!isFullySpecialized())
    throw std::runtime_error(
        "Kernel has unspecialized parameters; call execute(rawArgs) instead.");
  if (!resultInfo.hasResult()) {
    getJit().entryPoint();
    return {nullptr, 0};
  }
  // Allocate a result buffer on-the-fly.
  auto buf = std::make_unique<char[]>(resultInfo.bufferSize);
  std::vector<void *> rawArgs = {buf.get()};
  execute(rawArgs);
  return {buf.release(), resultInfo.bufferSize};
}

void (*cudaq::CompiledModule::JitArtifact::getEntryPoint() const)() {
  return entryPoint;
}

cudaq::JitEngine cudaq::CompiledModule::JitArtifact::getEngine() const {
  return engine;
}

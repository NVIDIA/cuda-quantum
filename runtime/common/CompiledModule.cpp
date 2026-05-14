/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledModule.h"
#include <string_view>

cudaq::FatQuakeModule::FatQuakeModule(std::string kernelName)
    : name(std::move(kernelName)) {}

cudaq::SourceModule::SourceModule(std::string kernelName, KernelThunkType fn)
    : FatQuakeModule(std::move(kernelName)) {
  addArtifact(name, FunctionPtrArtifact{fn});
}

cudaq::SourceModule::SourceModule(std::string kernelName,
                                  const void *mlirModuleOpaquePtr)
    : FatQuakeModule(std::move(kernelName)) {
  addArtifact(name, MlirArtifact{mlirModuleOpaquePtr, nullptr});
}

std::optional<cudaq::FatQuakeModule::JitArtifact>
cudaq::FatQuakeModule::getJit() const {
  return getJit(name);
}

std::optional<cudaq::FatQuakeModule::JitArtifact>
cudaq::FatQuakeModule::getJit(std::string_view jitName) const {
  auto *jit = artifacts.get<JitArtifact>(jitName);
  return jit ? std::optional<JitArtifact>{*jit} : std::nullopt;
}

std::optional<cudaq::FatQuakeModule::MlirArtifact>
cudaq::FatQuakeModule::getMlir() const {
  return getMlir(name);
}

std::optional<cudaq::FatQuakeModule::MlirArtifact>
cudaq::FatQuakeModule::getMlir(std::string_view mlirName) const {
  auto *mlir = artifacts.get<MlirArtifact>(mlirName);
  return mlir ? std::optional<MlirArtifact>{*mlir} : std::nullopt;
}

std::optional<cudaq::FatQuakeModule::FunctionPtrArtifact>
cudaq::FatQuakeModule::getFunctionPtr() const {
  return getFunctionPtr(name);
}

std::optional<cudaq::FatQuakeModule::FunctionPtrArtifact>
cudaq::FatQuakeModule::getFunctionPtr(std::string_view fnName) const {
  auto *fn = artifacts.get<FunctionPtrArtifact>(fnName);
  return fn ? std::optional<FunctionPtrArtifact>{*fn} : std::nullopt;
}

bool cudaq::FatQuakeModule::isFullySpecialized() const {
  return getArgsCreator() == nullptr;
}

int64_t (*cudaq::FatQuakeModule::getArgsCreator() const)(const void *,
                                                         void **) {
  auto jit = getJit(name + ".argsCreator");
  return jit ? reinterpret_cast<int64_t (*)(const void *, void **)>(jit->fn)
             : nullptr;
}

std::optional<std::int64_t> cudaq::FatQuakeModule::getReturnOffset() const {
  auto jit = getJit(name + ".returnOffset");
  if (!jit)
    return std::nullopt;
  auto fn = reinterpret_cast<std::int64_t (*)()>(jit->fn);
  return fn();
}

const cudaq::Resources *cudaq::FatQuakeModule::getResources() const {
  return getResources(name);
}

const cudaq::Resources *
cudaq::FatQuakeModule::getResources(std::string_view resourcesName) const {
  auto *res = artifacts.get<ResourcesArtifact>(resourcesName);
  return res ? &res->getResources() : nullptr;
}

void cudaq::FatQuakeModule::addArtifact(std::string name,
                                        CompiledArtifact artifact) {
  artifacts.add(std::move(name), std::move(artifact));
}

void (*cudaq::FatQuakeModule::JitArtifact::getFn() const)() { return fn; }

cudaq::JitEngine cudaq::FatQuakeModule::JitArtifact::getEngine() const {
  return engine;
}

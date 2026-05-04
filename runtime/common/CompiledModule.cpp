/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledModule.h"
#include <string_view>

cudaq::CompiledModule::CompiledModule(std::string kernelName)
    : name(std::move(kernelName)) {}

std::optional<cudaq::CompiledModule::JitArtifact>
cudaq::CompiledModule::getJit() const {
  return getJit(name);
}

std::optional<cudaq::CompiledModule::JitArtifact>
cudaq::CompiledModule::getJit(std::string_view jitName) const {
  auto *jit = artifacts.get<JitArtifact>(jitName);
  return jit ? std::optional<JitArtifact>{*jit} : std::nullopt;
}

std::optional<cudaq::CompiledModule::MlirArtifact>
cudaq::CompiledModule::getMlir() const {
  return getMlir(name);
}

std::optional<cudaq::CompiledModule::MlirArtifact>
cudaq::CompiledModule::getMlir(std::string_view mlirName) const {
  auto *mlir = artifacts.get<MlirArtifact>(mlirName);
  return mlir ? std::optional<MlirArtifact>{*mlir} : std::nullopt;
}

bool cudaq::CompiledModule::isFullySpecialized() const {
  return getArgsCreator() == nullptr;
}

int64_t (*cudaq::CompiledModule::getArgsCreator() const)(const void *,
                                                         void **) {
  auto jit = getJit(name + ".argsCreator");
  return jit ? reinterpret_cast<int64_t (*)(const void *, void **)>(jit->fn)
             : nullptr;
}

std::optional<std::int64_t> cudaq::CompiledModule::getReturnOffset() const {
  auto jit = getJit(name + ".returnOffset");
  if (!jit)
    return std::nullopt;
  auto fn = reinterpret_cast<std::int64_t (*)()>(jit->fn);
  return fn();
}

const cudaq::Resources *cudaq::CompiledModule::getResources() const {
  return getResources(name);
}

const cudaq::Resources *
cudaq::CompiledModule::getResources(std::string_view resourcesName) const {
  auto *res = artifacts.get<ResourcesArtifact>(resourcesName);
  return res ? &res->getResources() : nullptr;
}

void cudaq::CompiledModule::addArtifact(std::string name,
                                        CompiledArtifact artifact) {
  artifacts.add(std::move(name), std::move(artifact));
}

void (*cudaq::CompiledModule::JitArtifact::getFn() const)() { return fn; }

cudaq::JitEngine cudaq::CompiledModule::JitArtifact::getEngine() const {
  return engine;
}

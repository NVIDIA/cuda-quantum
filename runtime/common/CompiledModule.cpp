/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledModule.h"
#include <stdexcept>

cudaq::CompiledModule::CompiledModule(std::string kernelName)
    : name(std::move(kernelName)) {}

std::optional<cudaq::CompiledModule::JitArtifact>
cudaq::CompiledModule::getJit(std::optional<std::string> jitName) const {
  auto name = jitName.value_or(this->name);
  auto it = artifacts.find(name);
  if (it == artifacts.end())
    return std::nullopt;
  const auto *jit = std::get_if<JitArtifact>(&it->second);
  return jit ? std::optional(*jit) : std::nullopt;
}

std::optional<cudaq::CompiledModule::MlirArtifact>
cudaq::CompiledModule::getMlir(std::optional<std::string> mlirName) const {
  auto name = mlirName.value_or(this->name + ".mlir");
  auto it = artifacts.find(name);
  if (it == artifacts.end())
    return std::nullopt;
  const auto *mlir = std::get_if<MlirArtifact>(&it->second);
  return mlir ? std::optional(*mlir) : std::nullopt;
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

const cudaq::Resources *cudaq::CompiledModule::getResources(
    std::optional<std::string> resourcesName) const {
  auto name = resourcesName.value_or(this->name + ".resources");
  auto it = artifacts.find(name);
  if (it == artifacts.end())
    return nullptr;
  const auto *resources = std::get_if<ResourcesArtifact>(&it->second);
  return resources ? &resources->getResources() : nullptr;
}

void cudaq::CompiledModule::addArtifact(std::string name,
                                        CompiledArtifact artifact) {
  if (artifacts.contains(name))
    throw std::runtime_error("Artifact with name " + name + " already exists");
  artifacts.emplace(std::move(name), std::move(artifact));
}

void (*cudaq::CompiledModule::JitArtifact::getFn() const)() { return fn; }

cudaq::JitEngine cudaq::CompiledModule::JitArtifact::getEngine() const {
  return engine;
}

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq_internal/compiler/LayoutInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

using namespace mlir;

namespace cudaq_internal::compiler {

cudaq::ResultInfo CompiledModuleHelper::createResultInfo(Type resultTy,
                                                         bool isEntryPoint,
                                                         ModuleOp module) {
  cudaq::ResultInfo info;
  if (!resultTy || !isEntryPoint)
    return info;

  info.typeOpaquePtr = resultTy.getAsOpaquePointer();
  auto [size, offsets] = getResultBufferLayout(module, resultTy);
  info.bufferSize = size;
  info.fieldOffsets = std::move(offsets);
  return info;
}

std::vector<CompiledModuleHelper::NamedJitArtifact>
CompiledModuleHelper::createJitArtifacts(const std::string &kernelName,
                                         cudaq::JitEngine engine,
                                         const cudaq::ResultInfo &resultInfo,
                                         bool isFullySpecialized) {
  bool hasResult = resultInfo.hasResult();
  std::string fullName =
      std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName;
  std::string entryName =
      (hasResult || !isFullySpecialized) ? kernelName + ".thunk" : fullName;
  void (*entryPoint)() = engine.lookupRawNameOrFail(entryName);
  int64_t (*argsCreator)(const void *, void **) = nullptr;
  if (!isFullySpecialized)
    argsCreator = reinterpret_cast<int64_t (*)(const void *, void **)>(
        engine.lookupRawNameOrFail(kernelName + ".argsCreator"));

  std::vector<NamedJitArtifact> artifacts;
  artifacts.emplace_back(kernelName, cudaq::CompiledModule::JitArtifact{
                                         std::move(engine), entryPoint,
                                         argsCreator, std::nullopt});
  return artifacts;
}

cudaq::CompiledModule CompiledModuleHelper::createCompiledModule(
    std::string name, cudaq::ResultInfo resultInfo,
    std::vector<NamedJitArtifact> jitArtifacts) {
  return createCompiledModule(std::move(name), std::move(resultInfo),
                              std::move(jitArtifacts), {});
}

cudaq::CompiledModule CompiledModuleHelper::createCompiledModule(
    std::string name, cudaq::ResultInfo resultInfo,
    std::vector<NamedMlirArtifact> mlirArtifacts) {
  return createCompiledModule(std::move(name), std::move(resultInfo), {},
                              std::move(mlirArtifacts));
}

cudaq::CompiledModule CompiledModuleHelper::createCompiledModule(
    std::string name, cudaq::ResultInfo resultInfo,
    std::vector<NamedJitArtifact> jitArtifacts,
    std::vector<NamedMlirArtifact> mlirArtifacts) {
  cudaq::CompiledModule compiled(std::move(name));
  compiled.resultInfo = std::move(resultInfo);
  for (auto &[artName, artifact] : jitArtifacts)
    compiled.addArtifact(std::move(artName), std::move(artifact));
  for (auto &[artName, artifact] : mlirArtifacts)
    compiled.addArtifact(std::move(artName), std::move(artifact));
  return compiled;
}

} // namespace cudaq_internal::compiler

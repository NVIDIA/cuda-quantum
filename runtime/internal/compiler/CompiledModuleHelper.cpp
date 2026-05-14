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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using cudaq::CompiledModule;

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

std::vector<CompiledModuleHelper::NamedCompiledArtifact>
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

  std::vector<NamedCompiledArtifact> artifacts;
  artifacts.emplace_back(kernelName,
                         CompiledModule::JitArtifact{engine, entryPoint});
  if (!isFullySpecialized) {
    void (*argsCreatorFn)() =
        engine.lookupRawNameOrFail(kernelName + ".argsCreator");
    artifacts.emplace_back(kernelName + ".argsCreator",
                           CompiledModule::JitArtifact{engine, argsCreatorFn});
    if (hasResult) {
      void (*returnOffsetFn)() =
          engine.lookupRawNameOrFail(kernelName + ".returnOffset");
      artifacts.emplace_back(
          kernelName + ".returnOffset",
          CompiledModule::JitArtifact{engine, returnOffsetFn});
    }
  }
  return artifacts;
}

CompiledModuleHelper::NamedCompiledArtifact
CompiledModuleHelper::createResourcesArtifact(std::string name,
                                              cudaq::Resources rc) {
  return {std::move(name), CompiledModule::ResourcesArtifact{std::move(rc)}};
}

CompiledModuleHelper::NamedCompiledArtifact
CompiledModuleHelper::createMlirArtifact(std::string name, ModuleOp module,
                                         std::shared_ptr<MLIRContext> context) {
  const void *ptr = module.getAsOpaquePointer();
  return {std::move(name),
          CompiledModule::MlirArtifact{ptr, std::move(context)}};
}

ModuleOp CompiledModuleHelper::getMlirModuleOp(
    const CompiledModule::MlirArtifact &artifact) {
  return ModuleOp::getFromOpaquePointer(artifact.modulePtr);
}

CompiledModule CompiledModuleHelper::createCompiledModule(
    std::string name, cudaq::ResultInfo resultInfo,
    std::vector<NamedCompiledArtifact> compiledArtifacts,
    CompiledModule::CompilationMetadata metadata) {
  CompiledModule compiled(std::move(name));
  compiled.resultInfo = std::move(resultInfo);
  compiled.metadata = std::move(metadata);
  for (auto &[artName, artifact] : compiledArtifacts)
    compiled.addArtifact(std::move(artName), std::move(artifact));
  return compiled;
}

} // namespace cudaq_internal::compiler

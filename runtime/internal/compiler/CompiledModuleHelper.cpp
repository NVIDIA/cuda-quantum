/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "common/CompiledModule.h"
#include "common/KernelExecution.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LayoutInfo.h"
#include "nlohmann/json.hpp"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include <stdexcept>

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
CompiledModuleHelper::createCodegenArtifact(std::string name, std::string code,
                                            std::string outputNames) {
  return {std::move(name), CompiledModule::CodegenArtifact{
                               std::move(code), std::move(outputNames)}};
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

cudaq::FatQuakeModule::MlirArtifact
CompiledModuleHelper::loadMlirArtifact(const cudaq::SourceModule &module) {
  if (auto mlirArt = module.getMlir())
    return *mlirArt;

  auto [moduleOp, context] =
      cudaq_internal::compiler::Compiler::loadQuakeCodeByName(module.getName());
  cudaq::FatQuakeModule::MlirArtifact mlirArtifact(moduleOp,
                                                   std::move(context));
  return mlirArtifact;
}

std::vector<cudaq::KernelExecution>
CompiledModuleHelper::toKernelExecutions(const CompiledModule &compiled,
                                         bool requireTargetCode) {
  std::vector<cudaq::KernelExecution> codes;
  for (const auto &[name, mlirArtifact] : compiled.getMlirArtifacts()) {
    std::string code;
    nlohmann::json outputNames;
    if (auto targetCode = compiled.getTargetCode(name)) {
      code = targetCode->getCode();
      outputNames = nlohmann::json::parse(targetCode->getOutputNames());
    } else if (requireTargetCode) {
      throw std::runtime_error(
          "Cannot launch kernel '" + compiled.getName() +
          "': missing target code for MLIR artifact '" + name +
          "'. Compile the module with emitTargetCode=true and a non-empty "
          "codegen emission format before launching it on a remote QPU.");
    }

    std::optional<cudaq::JitEngine> jit;
    if (auto jitArtifact = compiled.getJit(name))
      jit = jitArtifact->getEngine();
    std::optional<cudaq::Resources> resourceCounts;
    if (auto *resources = compiled.getResources(name))
      resourceCounts = *resources;

    codes.emplace_back(name, std::move(code), std::move(jit),
                       std::move(resourceCounts), std::move(outputNames),
                       compiled.getMetadata().reorderIdx,
                       compiled.getMetadata().hasConditionalsOnMeasureResults);
  }
  return codes;
}

} // namespace cudaq_internal::compiler

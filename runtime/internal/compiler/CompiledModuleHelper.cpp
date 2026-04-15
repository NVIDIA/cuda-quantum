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

CompiledModuleBuilder::CompiledModuleBuilder(std::string kernelName)
    : compiled(std::move(kernelName)) {}

void CompiledModuleBuilder::setResultInfo(Type resultTy, bool isEntryPoint,
                                          ModuleOp module) {
  compiled.resultInfo = {};
  if (!resultTy || !isEntryPoint)
    return;

  compiled.resultInfo.typeOpaquePtr = resultTy.getAsOpaquePointer();
  auto [size, offsets] = getResultBufferLayout(module, resultTy);
  compiled.resultInfo.bufferSize = size;
  compiled.resultInfo.fieldOffsets = std::move(offsets);
}

void CompiledModuleBuilder::attachJit(cudaq::JitEngine engine,
                                      bool isFullySpecialized) {
  bool hasResult = compiled.resultInfo.hasResult();
  const std::string &name = compiled.name;
  std::string fullName = std::string(cudaq::runtime::cudaqGenPrefixName) + name;
  std::string entryName =
      (hasResult || !isFullySpecialized) ? name + ".thunk" : fullName;
  void (*entryPoint)() = engine.lookupRawNameOrFail(entryName);
  int64_t (*argsCreator)(const void *, void **) = nullptr;
  if (!isFullySpecialized)
    argsCreator = reinterpret_cast<int64_t (*)(const void *, void **)>(
        engine.lookupRawNameOrFail(name + ".argsCreator"));

  compiled.addArtifact(
      name, cudaq::CompiledModule::JitArtifact{std::move(engine), entryPoint,
                                               argsCreator, std::nullopt});
}

} // namespace cudaq_internal::compiler

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LayoutInfo.h"
#include "RuntimeMLIR.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/IR/DataLayout.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

using namespace mlir;

namespace {
cudaq::LayoutInfoType extractLayout(const std::string &kernelName,
                                    ModuleOp moduleOp) {
  auto *fnOp =
      moduleOp.lookupSymbol(cudaq::runtime::cudaqGenPrefixName + kernelName);
  if (!fnOp)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in the module.");
  // Extract layout information from the function's return type.
  Type returnTy = [&]() {
    if (fnOp->hasAttr(cudaq::runtime::enableCudaqRun)) {
      auto arrAttr =
          cast<ArrayAttr>(fnOp->getAttr(cudaq::runtime::enableCudaqRun));
      return cast<TypeAttr>(arrAttr[0]).getValue();
    }

    func::FuncOp kernelFunc = dyn_cast<func::FuncOp>(fnOp);
    if (!kernelFunc)
      throw std::runtime_error("expected a func::FuncOp.");
    if (kernelFunc.getResultTypes().size() == 0)
      throw std::runtime_error("function has no return type.");
    if (kernelFunc.getResultTypes().size() > 1)
      throw std::runtime_error("function has multiple return types.");
    return kernelFunc.getResultTypes()[0];
  }();

  returnTy = cudaq::opt::factory::convertToHostSideType(returnTy, moduleOp);

  auto attr = moduleOp->getAttr(cudaq::opt::factory::targetDataLayoutAttrName);
  if (!attr)
    throw std::runtime_error("module is malformed. missing data layout.");
  StringRef dataLayoutSpec = cast<StringAttr>(attr);
  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  CUDAQ_INFO("Data Layout: {}", dataLayout.getStringRepresentation());
  llvm::LLVMContext context;
  LLVMTypeConverter converter(fnOp->getContext());
  cudaq::opt::initializeTypeConversions(converter);
  auto structTy = dyn_cast<cudaq::cc::StructType>(returnTy);
  if (!structTy) {
    std::size_t totalSize = cudaq::opt::getDataSize(dataLayout, returnTy);
    if (totalSize == 0)
      throw std::runtime_error("size of result must not be 0.");
    return {totalSize, {}};
  }

  // Handle structure types
  auto llvmDialectTy = converter.convertType(structTy);
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto *llvmStructTy =
      cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  auto *layout = dataLayout.getStructLayout(llvmStructTy);
  std::size_t totalSize = layout->getSizeInBytes();
  std::size_t numElements = structTy.getMembers().size();
  std::vector<std::size_t> fieldOffsets;
  for (std::size_t i = 0; i < numElements; ++i)
    fieldOffsets.emplace_back(layout->getElementOffset(i));
  return {totalSize, fieldOffsets};
}

cudaq::LayoutInfoType extractLayout(const std::string &kernelName,
                                    const std::string &quakeCode) {
  auto moduleOp = parseSourceString<ModuleOp>(StringRef(quakeCode),
                                              cudaq::getMLIRContext());
  if (!moduleOp)
    throw std::runtime_error("module cannot be parsed");
  return extractLayout(kernelName, *moduleOp);
}
} // namespace

namespace cudaq {
LayoutInfoType getLayoutInfo(const std::string &name, void *opt_module) {
  if (opt_module) {
    // In Python, the interpreter already has the ModuleOp resident.
    ModuleOp mod{reinterpret_cast<Operation *>(opt_module)};
    return extractLayout(name, mod);
  }
  // In C++, the runtime has to reconstruct the ModuleOp.
  auto quakeCode = cudaq::get_quake_by_name(name, /*throwException=*/false);
  if (!quakeCode.empty())
    return extractLayout(name, quakeCode);
  return {};
}
} // namespace cudaq

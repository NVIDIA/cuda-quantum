/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LayoutExtractor.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "llvm/IR/DataLayout.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

using namespace llvm;
using namespace mlir;

mlir::MLIRContext *cudaq::LayoutExtractor::createContext() {
  mlir::DialectRegistry registry;
  cudaq::opt::registerCodeGenDialect(registry);
  cudaq::registerAllDialects(registry);
  auto context = new mlir::MLIRContext(registry);
  context->loadAllAvailableDialects();
  mlir::registerLLVMDialectTranslation(*context);
  return context;
}

std::pair<std::size_t, std::vector<std::size_t>>
cudaq::LayoutExtractor::extractLayout(const std::string &kernelName,
                                      const std::string &quakeCode) {
  std::unique_ptr<mlir::MLIRContext> mlirContext(createContext());
  auto m_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(quakeCode), mlirContext.get());
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");
  mlir::func::FuncOp kernelFunc;
  m_module->walk([&](mlir::func::FuncOp fOp) {
    if (fOp.getName().equals("__nvqpp__mlirgen__" + kernelName)) {
      kernelFunc = fOp;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in the module.");
  // Extract layout information from the function's return type
  std::size_t totalSize = 0;
  std::vector<std::size_t> fieldOffsets;
  // Only proceed if function has a return type
  if (kernelFunc.getNumResults() > 0) {
    mlir::Type returnType = kernelFunc.getResultTypes()[0];
    auto mod = kernelFunc->getParentOfType<ModuleOp>();
    StringRef dataLayoutSpec = "";
    if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    auto dataLayout = llvm::DataLayout(dataLayoutSpec);
    llvm::LLVMContext context;
    LLVMTypeConverter converter(kernelFunc.getContext());
    cudaq::opt::initializeTypeConversions(converter);
    // Handle structure types
    if (auto structType = mlir::dyn_cast<cc::StructType>(returnType)) {
      auto llvmDialectTy = converter.convertType(structType);
      LLVM::TypeToLLVMIRTranslator translator(context);
      auto *llvmStructTy =
          cast<llvm::StructType>(translator.translateType(llvmDialectTy));
      auto *layout = dataLayout.getStructLayout(llvmStructTy);
      totalSize = layout->getSizeInBytes();
      std::vector<std::size_t> fieldOffsets;
      for (std::size_t i = 0, I = structType.getMembers().size(); i != I; ++i)
        fieldOffsets.emplace_back(layout->getElementOffset(i));
    } else {
      // For non-struct types, just the size
      totalSize = cudaq::opt::getDataSize(dataLayout, returnType);
    }
  }
  return {totalSize, fieldOffsets};
}

std::pair<std::size_t, std::vector<std::size_t>>
cudaq::extractDataLayout(const std::string &kernelName,
                         const std::string &quakeCode) {
  cudaq::LayoutExtractor extractor;
  return extractor.extractLayout(kernelName, quakeCode);
}

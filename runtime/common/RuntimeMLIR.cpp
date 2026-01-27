/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeMLIR.h"
#include "ThunkInterface.h"
#include "common/FmtCore.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Support/TargetConfig.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"

using namespace mlir;

static llvm::StringMap<cudaq::Translation> &getTranslationRegistry() {
  static llvm::StringMap<cudaq::Translation> translationBundle;
  return translationBundle;
}

cudaq::Translation &cudaq::getTranslation(StringRef name) {
  auto namePair = name.split(':');
  auto &registry = getTranslationRegistry();
  if (!registry.count(namePair.first))
    throw std::runtime_error("Invalid IR Translation (" + namePair.first.str() +
                             ").");
  return registry[namePair.first];
}

static void
registerTranslation(StringRef name, StringRef description,
                    const cudaq::TranslateFromMLIRFunction &function) {
  assert(!name.contains(':') && "name and profile only");
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(function, description);
}

static void
registerTranslation(StringRef name, StringRef description,
                    const cudaq::TranslateFromMLIRFunctionExtended &f) {
  assert(!name.contains(':') && "name and profile only");
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(f &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(f, description);
}

cudaq::TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function) {
  registerTranslation(name, description, function);
}

cudaq::TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunctionExtended &function) {
  registerTranslation(name, description, function);
}

#include "RuntimeMLIRCommonImpl.h"

namespace {
std::once_flag mlir_init_flag;
MLIRContext *mlirContext;
std::unique_ptr<MLIRContext> createMLIRContext() {
  // Per-context initialization
  DialectRegistry registry;
  cudaq::opt::registerCodeGenDialect(registry);
  cudaq::registerAllDialects(registry);
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  registerLLVMDialectTranslation(*context);
  return context;
}
} // namespace

void cudaq::initializeMLIR() {
  // One-time initialization of LLVM/MLIR components
  std::call_once(mlir_init_flag, []() {
    cudaq::initializeLangMLIR();
    registerToQIRTranslation();
    registerToOpenQASMTranslation();
    registerToIQMJsonTranslation();

    mlirContext = createMLIRContext().release();
  });
}

MLIRContext *cudaq::getMLIRContext() {
  // One-time initialization of LLVM/MLIR components
  cudaq::initializeMLIR();
  return mlirContext;
}

std::unique_ptr<MLIRContext> cudaq::getOwningMLIRContext() {
  // One-time initialization of LLVM/MLIR components
  cudaq::initializeMLIR();
  return createMLIRContext();
}

std::optional<std::string>
cudaq::getEntryPointName(OwningOpRef<ModuleOp> &module) {
  for (auto &a : *module) {
    if (auto op = dyn_cast<mlir::func::FuncOp>(a)) {
      // Note: the .thunk function is where unmarshalling happens. It is *not*
      // an entry point.
      if (op.getName().endswith(".thunk"))
        return {op.getName().str()};
    }
  }
  return std::nullopt;
}

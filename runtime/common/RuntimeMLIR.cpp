/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/IQM/IQMJsonEmitter.h"
#include "cudaq/Target/OpenQASM/OpenQASMEmitter.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace cudaq {
static bool mlirLLVMInitialized = false;

static llvm::StringMap<cudaq::Translation> &getTranslationRegistry() {
  static llvm::StringMap<cudaq::Translation> translationBundle;
  return translationBundle;
}
cudaq::Translation &getTranslation(StringRef name) {
  auto &registry = getTranslationRegistry();
  if (!registry.count(name))
    throw std::runtime_error("Invalid IR Translation (" + name.str() + ").");
  return registry[name];
}

static void registerTranslation(StringRef name, StringRef description,
                                const TranslateFromMLIRFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(function, description);
}

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function) {
  registerTranslation(name, description, function);
}

void registerToQIRTranslation();
void registerToOpenQASMTranslation();
void registerToIQMJsonTranslation();

std::unique_ptr<MLIRContext> initializeMLIR() {
  if (!mlirLLVMInitialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    registerAllPasses();
    cudaq::opt::registerOptCodeGenPasses();
    cudaq::opt::registerOptTransformsPasses();
    registerToQIRTranslation();
    registerToOpenQASMTranslation();
    registerToIQMJsonTranslation();
    mlirLLVMInitialized = true;
  }

  // if (!llvmContext)
  //   llvmContext = std::make_unique<llvm::LLVMContext>();

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, AffineDialect, LLVM::LLVMDialect,
                  quake::QuakeDialect, cc::CCDialect, func::FuncDialect>();
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  registerLLVMDialectTranslation(*context);
  return context;
}

bool setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return false;

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine)
    return false;

  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  return true;
}

void optimizeLLVM(llvm::Module *module) {
  bool enableOpt = true;
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(module))
    throw std::runtime_error("Failed to optimize LLVM IR ");
}

void registerToQIRTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "qir", "translate from quake to qir adaptive",
      [](Operation *op, raw_ostream &output) {
        auto context = op->getContext();
        PassManager pm(context);
        std::string errMsg;
        llvm::raw_string_ostream errOs(errMsg);
        std::string qirBasePipelineConfig = "quake-to-qir,qir-to-base-qir-prep,"
                                            "llvm.func(quake-to-base-qir-func),"
                                            "qir-to-base-qir";
        if (failed(parsePassPipeline(qirBasePipelineConfig, pm, errOs)))
          return failure();
        if (failed(pm.run(op)))
          return failure();

        std::unique_ptr<llvm::LLVMContext> llvmContext =
            std::make_unique<llvm::LLVMContext>();
        llvmContext->setOpaquePointers(false);
        auto llvmModule = translateModuleToLLVMIR(op, *llvmContext);
        cudaq::optimizeLLVM(llvmModule.get());
        if (!cudaq::setupTargetTriple(llvmModule.get()))
          throw std::runtime_error(
              "Failed to setup the llvm module target triple.");

        // Map the LLVM Module to Bitcode that can be submitted
        llvm::SmallString<1024> bitCodeMem;
        llvm::raw_svector_ostream os(bitCodeMem);
        llvm::WriteBitcodeToFile(*llvmModule, os);
        output << llvm::encodeBase64(bitCodeMem.str());
        return success();
      });
}

void registerToOpenQASMTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "qasm2", "translate from quake to openQASM 2.0",
      [](Operation *op, raw_ostream &output) {
        PassManager pm(op->getContext());
        if (failed(pm.run(op)))
          throw std::runtime_error("Lowering failed.");
        return cudaq::translateToOpenQASM(op, output);
      });
}

void registerToIQMJsonTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "iqm", "translate from quake to IQM's json format",
      [](Operation *op, raw_ostream &output) {
        return cudaq::translateToIQMJson(op, output);
      });
}
} // namespace cudaq

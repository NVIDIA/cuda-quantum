/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "TargetBackend.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <iostream>

using namespace mlir;

namespace cudaq {

const static std::string BOLD = "\033[1m";
const static std::string RED = "\033[91m";
const static std::string BLUE = "\033[94m";
const static std::string CLEAR = "\033[0m";

bool TargetBackend::setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "NO target: " << errorMessage << "\n";
    return false;
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine) {
    llvm::errs() << "Unable to create target machine\n";
    return false;
  }
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  return true;
}

std::unique_ptr<llvm::Module>
TargetBackend::compile(MLIRContext &context, const std::string_view quakeCode) {
  auto m_module = parseSourceString<ModuleOp>(quakeCode, &context);
  DiagnosticEngine &engine = context.getDiagEngine();
  engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
    std::cout << BOLD << RED << "[nvqpp-mlir] Dumping Module after error.\n"
              << CLEAR;
    for (auto &n : diag.getNotes()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      n.print(os);
      os.flush();
      std::cout << BOLD << RED << "[nvqpp-mlir] Reported Error: " << s << "\n"
                << CLEAR;
    }
    bool should_propagate_diagnostic = true;
    return failure(should_propagate_diagnostic);
  });

  PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  pm.addPass(createInlinerPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  auto loop_unroller = createLoopUnrollPass(
      /*unrollFactor*/ -1, /*unrollUpToFactor*/ false, /*unrollFull*/ true);
  pm.addNestedPass<func::FuncOp>(std::move(loop_unroller));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createConvertToQIRPass());

  if (failed(pm.run(*m_module))) {
    llvm::errs() << "MLIRRuntime pass pipeline failed!\n";
    return nullptr;
  }

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvmContext.setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(m_module.get(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return nullptr;
  }

  // Initialize LLVM targets.
  /// Optionally run an optimization pipeline over the llvm module.
  bool enableOpt = true;
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get()))
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";

  if (!setupTargetTriple(llvmModule.get())) {
    llvm::errs() << "Failed to setup the llvm module target triple.\n";
    return nullptr;
  }

  return llvmModule;
}

/// Pointer to the global MLIR Context
extern std::unique_ptr<MLIRContext> mlirContext;

std::unique_ptr<llvm::Module>
TargetBackend::lowerQuakeToBaseProfile(Kernel &thunk,
                                       llvm::LLVMContext &localLLVMContext,
                                       cudaq::spin_op *term, void *kernelArgs) {
  MLIRContext *localContext = mlirContext.get();
  auto quakeCode = thunk.getQuakeCode();

  auto m_module = parseSourceString<ModuleOp>(quakeCode, localContext);
  DiagnosticEngine &engine = localContext->getDiagEngine();
  engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
    std::cout << "[qpud-mlir] Dumping Module after error.\n";
    for (auto &n : diag.getNotes()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      n.print(os);
      os.flush();
      std::cout << "[qpud-mlir] Reported Error: " << s << "\n";
    }
    bool should_propagate_diagnostic = true;
    return failure(should_propagate_diagnostic);
  });

  // remove the thunk function
  std::string kernelName(thunk.name());
  auto thunkFunction =
      m_module->lookupSymbol<func::FuncOp>(kernelName + ".thunk");
  if (thunkFunction)
    thunkFunction->erase();

  // Synthesize the Quake code and lower to QIR
  PassManager pm(localContext);
  // Only synthesize if we have runtime args
  if (kernelArgs)
    pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, kernelArgs));
  pm.addPass(createInlinerPass());
  applyNativeGateSetPasses(pm);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createLoopUnrollPass(-1, false, true));
  if (term) {
    // add a pass that adds measures.
    auto binarySymplecticForm = term->get_bsf()[0];
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createQuakeObserveAnsatzPass(binarySymplecticForm));
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(cudaq::opt::createConvertToQIRPass());
  cudaq::opt::addBaseProfilePipeline(pm);

  if (failed(pm.run(*m_module))) {
    llvm::errs() << "Failed to run the MLIR Pass Manager.\n";
    return nullptr;
  }

  auto llvmModule = translateModuleToLLVMIR(m_module.get(), localLLVMContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return nullptr;
  }

  // Initialize LLVM targets.
  /// Optionally run an optimization pipeline over the llvm module.
  bool enableOpt = true;
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  if (!setupTargetTriple(llvmModule.get())) {
    llvm::errs() << "Failed to setup the llvm module target triple.\n";
    return nullptr;
  }

  return llvmModule;
}
} // namespace cudaq

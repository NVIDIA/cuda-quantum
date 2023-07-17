/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/IQM/IQMJsonEmitter.h"
#include "cudaq/Target/OpenQASM/OpenQASMEmitter.h"
#include "cudaq/Todo.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional,
                  llvm::cl::desc("<input quake mlir file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<unsigned> optLevel(
    "opt-level",
    llvm::cl::desc(
        "Set the LLVM optimization level. Default is 3. Use 0 to disable."),
    llvm::cl::value_desc("level"), llvm::cl::init(3));

static llvm::cl::opt<unsigned> sizeLevel(
    "size-level",
    llvm::cl::desc("Set the LLVM size optimization level. Default is 0."),
    llvm::cl::value_desc("level"), llvm::cl::init(0));

static llvm::cl::opt<std::string> convertTo(
    "convert-to",
    llvm::cl::desc(
        "Specify the translation output to be created. [Default: \"qir\"]"),
    llvm::cl::value_desc("target dialect [\"qir\", \"qir-base\", "
                         "\"openqasm\", \"iqm\"]"),
    llvm::cl::init("qir"));

static llvm::cl::opt<bool> emitLLVM(
    "emit-llvm",
    llvm::cl::desc("Emit LLVM IR as the output. If set to false, the "
                   "translation will terminate with the selected dialect."),
    llvm::cl::init(true));

constexpr static char BOLD[] = "\033[1m";
constexpr static char RED[] = "\033[91m";
constexpr static char CLEAR[] = "\033[0m";

using namespace mlir;

// Pipeline builder to convert Quake to QIR.
template <bool BaseProfile = false>
void addPipelineToQIR(PassManager &pm) {
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(cudaq::opt::createPromoteRefToVeqAlloc());
  pm.addPass(cudaq::opt::createConvertToQIRPass());
  pm.addPass(createCanonicalizerPass());
  if constexpr (BaseProfile) {
    cudaq::opt::addBaseProfilePipeline(pm);
  }
}

static void checkErrorCode(const std::error_code &ec) {
  if (ec) {
    llvm::errs() << "could not open output file";
    std::exit(ec.value());
  }
}

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerTranslationCLOptions();
  registerAllPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "quake mlir to llvm ir compiler\n");

  DialectRegistry registry;
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  registerAllDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError())
    cudaq::emitFatalError(UnknownLoc::get(&context),
                          "Could not open input file: " + ec.message());

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  DiagnosticEngine &engine = context.getDiagEngine();
  engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
    llvm::errs() << BOLD << RED
                 << "[quake-translate] Dumping Module after error.\n"
                 << CLEAR;
    for (auto &n : diag.getNotes()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      n.print(os);
      os.flush();
      llvm::errs() << BOLD << RED << "[quake-translate] Reported Error: " << s
                   << "\n"
                   << CLEAR;
    }
    bool should_propagate_diagnostic = true;
    return failure(should_propagate_diagnostic);
  });

  PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Some translations do not involve translation to LLVM IR. These translations
  // are done directly from the MLIR Module to an output file.
  llvm::Optional<std::function<LogicalResult(Operation *, raw_ostream &)>>
      directTranslation;
  llvm::StringSwitch<std::function<void()>>(convertTo)
      .Case("qir", [&]() { addPipelineToQIR<>(pm); })
      .Case("qir-base", [&]() { addPipelineToQIR</*baseProfile=*/true>(pm); })
      .Case("openqasm",
            [&]() { directTranslation = cudaq::translateToOpenQASM; })
      .Case("iqm", [&]() { directTranslation = cudaq::translateToIQMJson; })
      .Default([]() {})();

  std::error_code ec;
  llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
  checkErrorCode(ec);

  if (directTranslation) {
    // The translation pass will output directly to the output file. It will
    // never use the PassManager.
    if (failed((*directTranslation)(module->getOperation(), out.os()))) {
      cudaq::emitFatalError(module->getLoc(), "translation failed");
      return 1;
    }
    out.keep();
    return 0;
  }

  if (failed(pm.run(*module)))
    cudaq::emitFatalError(module->getLoc(), "pipeline failed");

  if (!emitLLVM) {
    out.os() << *module << '\n';
    out.keep();
    return 0;
  }

  // Register the translation to LLVM IR with the MLIR context.
  registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  llvmContext.setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmModule)
    cudaq::emitFatalError(module->getLoc(), "Failed to emit LLVM IR");

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  ExecutionEngine::setupTargetTriple(llvmModule.get());

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = makeOptimizingTransformer(optLevel, sizeLevel,
                                               /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << '\n';
    std::exit(1);
  }

  // Output the LLVM IR to the output file.
  if (ec)
    cudaq::emitFatalError(module->getLoc(),
                          "Failed to open output file '" + outputFilename);

  out.os() << *llvmModule << "\n";
  out.keep();
  return 0;
}

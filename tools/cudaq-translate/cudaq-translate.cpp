/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Support/Version.h"
#include "cudaq/Todo.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
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

static void checkErrorCode(const std::error_code &ec) {
  if (ec) {
    llvm::errs() << "could not open output file";
    std::exit(ec.value());
  }
}

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);

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

  std::error_code ec;
  llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
  checkErrorCode(ec);
  llvm::function_ref<void()> targetAction = [&]() {
    out.os() << *module << '\n';
  };
  bool targetUsesLlvm = emitLLVM;
  auto *modOp = module->getOperation();
  auto modLoc = module->getLoc();
  // Declare actions here to avoid outer closure going out of scope below.
  auto iqmAction = [&]() {
    if (failed(cudaq::translateToIQMJson(modOp, out.os()))) {
      cudaq::emitFatalError(modLoc, "translation failed");
      std::exit(1);
    }
  };
  auto qasmAction = [&]() {
    if (failed(cudaq::translateToOpenQASM(modOp, out.os()))) {
      cudaq::emitFatalError(modLoc, "translation failed");
      std::exit(1);
    }
  };

  llvm::StringSwitch<std::function<void()>>(convertTo)
      .Case("qir", [&]() { cudaq::opt::addPipelineToQIR<>(pm); })
      .Case("qir-base",
            [&]() { cudaq::opt::addPipelineToQIR</*baseProfile=*/true>(pm); })
      .Case("openqasm",
            [&]() {
              targetUsesLlvm = false;
              cudaq::opt::addPipelineToOpenQASM(pm);
              targetAction = qasmAction;
            })
      .Case("iqm",
            [&]() {
              targetUsesLlvm = false;
              cudaq::opt::addPipelineToIQMJson(pm);
              targetAction = iqmAction;
            })
      .Default([]() {})();

  if (failed(pm.run(*module)))
    cudaq::emitFatalError(module->getLoc(), "pipeline failed");

  if (!targetUsesLlvm) {
    targetAction();
    out.keep();
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Everything from here down handles the cases where code generation uses LLVM
  // to generate the code.

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

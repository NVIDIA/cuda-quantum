/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Common/InlinerInterface.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/Version.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace llvm;

/// Dialect extension to allow inlining of the MLIR defined LLVM-IR dialects
/// which lacks inlining support out of the box.
class InlinerExtension
    : public mlir::DialectExtension<InlinerExtension, mlir::LLVM::LLVMDialect> {
public:
  void apply(mlir::MLIRContext *ctx,
             mlir::LLVM::LLVMDialect *dialect) const override {
    dialect->addInterfaces<cudaq::EnableInlinerInterface>();
    ctx->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  };
};

static void registerInlinerExtension(mlir::DialectRegistry &registry) {
  registry.addExtensions<InlinerExtension>();
}

/// @brief Add a command line flag for loading plugins
static cl::list<std::string> CudaQPlugins(
    "load-cudaq-plugin",
    cl::desc("Load CUDA Quantum plugin by specifying its library"));

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);

  mlir::registerAllPasses();
  cudaq::opt::registerOptCodeGenPasses();
  cudaq::opt::registerOptTransformsPasses();
  cudaq::opt::registerAggressiveEarlyInlining();
  cudaq::opt::registerUnrollingPipeline();
  cudaq::opt::registerBaseProfilePipeline();
  cudaq::opt::registerTargetPipelines();

  // See if we have been asked to load a pass plugin,
  // if so load it.
  std::vector<std::string> args(&argv[0], &argv[0] + argc);
  for (std::size_t i = 0; i < args.size(); i++) {
    if (args[i].find("-load-cudaq-plugin") != std::string::npos) {
      auto Plugin = cudaq::Plugin::Load(args[i + 1]);
      if (!Plugin) {
        errs() << "Failed to load passes from '" << args[i + 1]
               << "'. Request ignored.\n";
        return 1;
      }
      Plugin.get().registerExtensions();
      i++;
    }
  }

  mlir::DialectRegistry registry;
  registry.insert<quake::QuakeDialect, cudaq::cc::CCDialect>();
  registerAllDialects(registry);
  registerInlinerExtension(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nvq++ optimizer\n", registry));
}

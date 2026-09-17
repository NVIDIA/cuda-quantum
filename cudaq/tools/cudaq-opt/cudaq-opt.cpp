/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/CodeGenDialect.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/InlinerInterface.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/Version.h"
#include "cudaq/Target/TargetDatabase.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
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
static cl::list<std::string>
    CudaQPlugins("load-cudaq-plugin",
                 cl::desc("Load CUDA-Q plugin by specifying its library"));

/// Registers `pipelineText` as a named pipeline, `pipelineName`. Once
/// registered, nvq++ can select it by referencing `pipelineName` inside an
/// ordinary
/// `-pass-pipeline='builtin.module(<pipelineName>)'` argument.
static void registerTargetPassPipeline(const std::string &pipelineName,
                                       const std::string &targetName,
                                       std::string pipelineText) {
  mlir::PassPipelineRegistration<>(
      pipelineName,
      "Precompiled pass pipeline for target '" + targetName + "'.",
      [pipelineName, pipelineText](mlir::OpPassManager &pm) {
        if (failed(mlir::parsePassPipeline(pipelineText, pm))) {
          const std::string message = "failed to parse the pass pipeline "
                                      "configured for target '" +
                                      pipelineName + "'";
          llvm::report_fatal_error(llvm::StringRef(message));
        }
      });
}

/// Registers a named pipeline for every in-tree target (and, within it,
/// every configuration-matrix entry) that configures a `TargetPassPipeline`.
static void registerAllTargetPassPipelines() {
  for (const auto &[name, config] : cudaq::config::listBuiltinTargets()) {
    const std::string targetName(name);
    if (config->BackendConfig.has_value() &&
        !config->BackendConfig->TargetPassPipeline.empty())
      registerTargetPassPipeline("target-pass-pipeline-" + targetName,
                                 targetName,
                                 config->BackendConfig->TargetPassPipeline);
    for (const auto &entry : config->ConfigMap)
      if (!entry.Config.TargetPassPipeline.empty())
        registerTargetPassPipeline("target-pass-pipeline-" + targetName + "-" +
                                       entry.Name,
                                   targetName, entry.Config.TargetPassPipeline);
  }
}

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);

  cudaq::registerAllCLOptions();
  cudaq::registerAllPasses();
  // Every target's TargetPassPipeline becomes its own named, registered
  // pipeline.
  registerAllTargetPassPipelines();

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
  cudaq::registerAllDialects(registry);
  registry.insert<cudaq::codegen::CodeGenDialect>();
  registerInlinerExtension(registry);
  mlir::func::registerInlinerExtension(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nvq++ optimizer\n", registry));
}

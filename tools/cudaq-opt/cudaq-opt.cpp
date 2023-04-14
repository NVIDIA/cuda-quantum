/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Common/InlinerInterface.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
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

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  cudaq::opt::registerOptCodeGenPasses();
  cudaq::opt::registerOptTransformsPasses();
  cudaq::opt::registerTargetPipelines();

  mlir::DialectRegistry registry;
  registry.insert<quake::QuakeDialect, cudaq::cc::CCDialect>();
  registerAllDialects(registry);
  registerInlinerExtension(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nvq++ optimizer\n", registry));
}

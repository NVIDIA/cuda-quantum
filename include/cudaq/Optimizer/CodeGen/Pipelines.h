/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file
/// Define some pipeline instantiation functions that can be shared between
/// the various tools and the runtime.

#pragma once

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {

/// @brief Pipeline builder to convert Quake to QIR. `convertTo` should be
/// specified if `QIRProfile` is true.
/// @param QIRProfile whether or not this is lowering to a specific QIR profile
/// @param pm Pass manager to append passes to
/// @param convertTo String name of QIR profile (e.g., `qir-base`,
/// `qir-adaptive`)
template <bool QIRProfile = false>
void addPipelineToQIR(mlir::PassManager &pm,
                      llvm::StringRef convertTo = "none") {
  pm.addNestedPass<mlir::func::FuncOp>(
      cudaq::opt::createApplyControlNegations());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addPass(cudaq::opt::createLoopNormalize());
  cudaq::opt::LoopUnrollOptions luo;
  luo.allowBreak = convertTo.equals("qir-adaptive");
  pm.addPass(cudaq::opt::createLoopUnroll(luo));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (convertTo.equals("qir-base"))
    pm.addNestedPass<mlir::func::FuncOp>(
        cudaq::opt::createDelayMeasurementsPass());
  pm.addPass(cudaq::opt::createConvertToQIRPass());
  if constexpr (QIRProfile) {
    cudaq::opt::addQIRProfilePipeline(pm, convertTo);
  }
}

inline void addPipelineToOpenQASM(mlir::PassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

inline void addPipelineToIQMJson(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<mlir::func::FuncOp>(
      cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}
} // namespace cudaq::opt

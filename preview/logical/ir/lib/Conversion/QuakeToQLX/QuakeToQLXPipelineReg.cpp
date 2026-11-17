/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qlx/Conversion/QuakeToQLXPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace qlx {

void buildPrepareQuakeForQLXPipeline(mlir::OpPassManager &pm) {
  pm.addPass(qlx::createEraseQuakeGlobalPhase());
  pm.addPass(qlx::createPruneDeadCCLoopCarries());
}

void buildQuakeToQLXPipeline(mlir::OpPassManager &pm,
                             const QuakeToQLXPipelineOptions &options) {
  buildPrepareQuakeForQLXPipeline(pm);
  ConvertQuakeToQLXOptions convertOptions;
  convertOptions.entryPoint = options.entryPoint;
  pm.addPass(qlx::createConvertQuakeToQLX(convertOptions));
}

void registerQuakeToQLXPipelines() {
  static bool registered = []() {
    mlir::PassPipelineRegistration<>(
        "prepare-quake-for-qlx",
        "Finalize CUDA-Q Quake already lowered through "
        "convert-to-linear-values for QLX P0 conversion",
        buildPrepareQuakeForQLXPipeline);
    mlir::PassPipelineRegistration<QuakeToQLXPipelineOptions>(
        "quake-to-qlx-pipeline",
        "Convert CUDA-Q Quake already lowered through "
        "convert-to-linear-values to QLX P0: runs prepare-quake-for-qlx, then "
        "convert-quake-to-qlx",
        buildQuakeToQLXPipeline);
    return true;
  }();
  (void)registered;
}

} // namespace qlx

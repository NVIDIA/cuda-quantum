/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Conversion/QuakeToQLXPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace qlx {

void registerPrepareQuakeForQLXPipeline() {
  static bool registered = []() {
    mlir::PassPipelineRegistration<>(
        "prepare-quake-for-qlx",
        "Finalize CUDA-Q Quake already lowered through "
        "convert-to-linear-values for QLX P0 conversion",
        [](mlir::OpPassManager &pm) {
          pm.addPass(qlx::createPruneDeadCCLoopCarries());
        });
    return true;
  }();
  (void)registered;
}

} // namespace qlx

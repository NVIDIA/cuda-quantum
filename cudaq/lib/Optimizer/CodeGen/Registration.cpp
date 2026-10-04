/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/InitAllPasses.h"
#include <mutex>

void cudaq::registerAllPasses() {
  // Compiler bindings and the execution runtime can initialize independently.
  static std::once_flag registered;
  std::call_once(registered, [] {
    // General MLIR passes
    mlir::registerTransformsPasses();

    // All the CUDA-Q passes and pipelines.
    registerCudaqPassesAndPipelines();
  });
}

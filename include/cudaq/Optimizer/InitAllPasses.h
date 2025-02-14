/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq {

inline void registerCudaqPassesAndPipelines() {
  // CUDA-Q passes
  opt::registerOptCodeGenPasses();
  opt::registerOptTransformsPasses();

  // CUDA-Q pipelines
  opt::registerAggressiveEarlyInliningPipeline();
  opt::registerUnrollingPipeline();
  opt::registerClassicalOptimizationPipeline();
  opt::registerToExecutionManagerCCPipeline();
  opt::registerToQIRAPIPipeline();
  opt::registerTargetPipelines();
  opt::registerWireSetToProfileQIRPipeline();
  opt::registerMappingPipeline();
}

inline void registerAllPasses() {
  // General MLIR passes
  mlir::registerTransformsPasses();

  // All the CUDA-Q passes and pipelines.
  registerCudaqPassesAndPipelines();
}

} // namespace cudaq

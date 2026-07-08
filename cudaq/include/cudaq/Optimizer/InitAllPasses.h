/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq {

inline void registerCudaqPassesAndPipelines() {
  // CUDA-Q passes
  opt::registerOptCodeGenPasses();
  opt::registerOptTransformsPasses();

  // CUDA-Q pipelines
  opt::registerAggressiveInliningPipeline();
  opt::registerUnrollingPipeline();
  opt::registerPhaseFoldingPipeline();
  opt::registerClassicalOptimizationPipeline();
  opt::registerToExecutionManagerCCPipeline();
  opt::registerToQIRAPIPipeline();
  opt::registerTargetPipelines();
  opt::registerWireSetToProfileQIRPipeline();
  opt::registerMappingPipeline();
  opt::registerToCFGPipeline();

  // JIT compiler pipelines
  opt::registerJITPipelines();

  // AOT compiler pipelines
  opt::registerAOTPipelines();
}

inline void registerAllPasses() {
  // General MLIR passes
  mlir::registerTransformsPasses();

  // All the CUDA-Q passes and pipelines.
  registerCudaqPassesAndPipelines();
}

inline void registerAllCLOptions() {
  opt::builder::registerCUDAQBuilderCLOptions();
}
} // namespace cudaq

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

inline void registerAllPasses() {
  // General MLIR passes
  mlir::registerTransformsPasses();

  // NVQPP passes
  opt::registerOptCodeGenPasses();
  opt::registerOptTransformsPasses();
  opt::registerAggressiveEarlyInlining();

  // Pipelines
  opt::registerUnrollingPipeline();
  opt::registerToExecutionManagerCCPipeline();
  opt::registerTargetPipelines();
  opt::registerWireSetToProfileQIRPipeline();
  opt::registerMappingPipeline();
}

} // namespace cudaq

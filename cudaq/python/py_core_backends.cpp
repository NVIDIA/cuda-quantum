/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "runtime/common/py_EstimateResult.h"
#include "runtime/common/py_Resources.h"
#include "runtime/cudaq/target/py_compile_target.h"
#include "runtime/cudaq/trace/py_trace.h"

NB_MODULE(_backends, m) {
  cudaq::bindCompileTarget(m);
  cudaq::bindResources(m);
  cudaq::bindEstimateResult(m);
  // Tracing uses the shared logger, not the execution runtime.
  cudaq::bindTrace(m);
}

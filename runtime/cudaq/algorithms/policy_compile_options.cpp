/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "cudaq/algorithms/policies.h"

// These are all "hidden friends", meaning that they are not declared anywhere
// outside of friend declarations within policy structs. This means they can
// only be discovered through ADL, using `cudaq::get_compile_options`.

namespace cudaq {

CompileOptions get_compile_options_impl(const other_policies &policy) {
  const auto *ctx = cudaq::getExecutionContext();
  CompileOptions opts;
  opts.emitResourceCounts = ctx && ctx->name == "resource-count";
  opts.disableQuantumOpts = ctx && ctx->name == "tracer";
  return opts;
}

CompileOptions get_compile_options_impl(const sample_policy &policy) {
  CompileOptions opts;
  opts.storeReorderIdx = true;
  // `sample` does not support conditionals on measurement results.
  opts.failOnConditionalsOnMeasureResults = true;
  // TODO: we would like to set this to true, but local simulators currently
  // don't work with this flag
  // opts.addMeasurements = true;
  return opts;
}

CompileOptions get_compile_options_impl(const dem_policy &policy) {
  CompileOptions opts;
  opts.emitJit = true;
  opts.emitTargetCode = false;
  opts.skipTargetLoweringPipeline = true;
  return opts;
}

} // namespace cudaq

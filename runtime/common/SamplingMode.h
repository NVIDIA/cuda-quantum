/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExecutionContext.h"

namespace cudaq::details {

enum class SamplingMode {
  // Standard sampling mode, measurements are ordered by the qubit allocation
  // order, no duplicates
  Default,
  // Explicit measurement mode, measurements are concatenated by the execution
  // order, duplicates allowed
  Explicit,
};

/// Derive sampling mode from execution context
inline SamplingMode getSamplingMode(const ExecutionContext *ctx) {
  if (!ctx)
    throw std::runtime_error("ExecutionContext is null.");
  if (ctx->name == "sample")
    return SamplingMode::Default;
  if (ctx->name == "sample_explicit")
    return SamplingMode::Explicit;
  throw std::runtime_error("Unknown sampling mode - " + ctx->name);
}

/// Check if this is any sampling context
inline bool isSamplingContext(const ExecutionContext *ctx) {
  if (!ctx)
    throw std::runtime_error("ExecutionContext is null.");
  return ctx->name.find("sample") != std::string::npos;
}

/// Check if this is explicit measurements sampling mode
inline bool isExplicitSamplingMode(const ExecutionContext *ctx) {
  if (!ctx)
    throw std::runtime_error("ExecutionContext is null.");
  return getSamplingMode(ctx) == SamplingMode::Explicit;
}

/// Derive sampling mode from execution context name
inline SamplingMode getSamplingMode(const std::string &contextName) {
  if (contextName == "sample")
    return SamplingMode::Default;
  if (contextName == "sample_explicit")
    return SamplingMode::Explicit;
  throw std::runtime_error("Unknown sampling mode - " + contextName);
}

} // namespace cudaq::details

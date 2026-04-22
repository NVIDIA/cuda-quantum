/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ExecutionContext.h"

namespace cudaq_internal::compiler {

/// Options for `Compiler::runPassPipeline` and `Compiler::lowerQuakeCode`.
struct CompileOptions {
  /// Build options from an execution context.
  static CompileOptions fromExecutionContext(const cudaq::ExecutionContext *ctx,
                                             bool emulate = false);
};

} // namespace cudaq_internal::compiler

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq_internal/compiler/CompileOptions.h"

cudaq_internal::compiler::CompileOptions
cudaq_internal::compiler::CompileOptions::fromExecutionContext(
    const cudaq::ExecutionContext *ctx, bool emulate) {
  (void)ctx;
  (void)emulate;
  return {};
}

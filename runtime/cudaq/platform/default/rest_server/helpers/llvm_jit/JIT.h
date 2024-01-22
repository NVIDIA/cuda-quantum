/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace cudaq {
/// Util to invoke a wrapped kernel defined by LLVM IR with serialized
/// arguments.
// We don't use `mlir::ExecutionEngine` because:
//  (1) we need to `setAutoClaimResponsibilityForObjectSymbols(true)` to work
//  around an assert bug ("Resolving symbol outside this responsibility set").
//  (2) skipping unnecessary `packFunctionArguments`.
void invokeWrappedKernel(std::string_view llvmIr, const std::string &kernelName,
                         void *args, std::uint64_t argsSize);
} // namespace cudaq

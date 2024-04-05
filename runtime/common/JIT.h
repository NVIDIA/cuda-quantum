/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace cudaq {
/// Util to invoke a wrapped kernel defined by LLVM IR with serialized
/// arguments.
// We don't use `mlir::ExecutionEngine` because:
//  (1) we need to `setAutoClaimResponsibilityForObjectSymbols(true)` to work
//  around an assert bug ("Resolving symbol outside this responsibility set").
//  (2) skipping unnecessary `packFunctionArguments`.
// Optionally, the JIT'ed kernel can be executed a number of times along with a
// post-execution callback. For example, sample a dynamic kernel.
void invokeWrappedKernel(
    std::string_view llvmIr, const std::string &kernelName, void *args,
    std::uint64_t argsSize, std::size_t numTimes = 1,
    std::function<void(std::size_t)> postExecCallback = {});
} // namespace cudaq

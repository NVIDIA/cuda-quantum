/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cudaq {
/// Util to invoke a wrapped kernel defined by LLVM IR with serialized
/// arguments.
// Note: We don't use `mlir::ExecutionEngine` to skip unnecessary
// `packFunctionArguments` (slow for raw LLVM IR containing many functions from
// included headers).
// Optionally, the JIT'ed kernel can be executed a number of
// times along with a post-execution callback. For example, sample a dynamic
// kernel.
std::unique_ptr<llvm::orc::LLJIT>
invokeWrappedKernel(std::string_view llvmIr, const std::string &kernelName,
                    void *args, std::uint64_t argsSize,
                    std::size_t numTimes = 1,
                    std::function<void(std::size_t)> postExecCallback = {});
} // namespace cudaq

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace llvm::orc {
class LLJIT;
}

namespace cudaq {

/// Util to create a wrapped kernel defined by LLVM IR with serialized
/// arguments.
// Note: We don't use `mlir::ExecutionEngine` to skip unnecessary
// `packFunctionArguments` (slow for raw LLVM IR containing many functions from
// included headers).
std::tuple<std::unique_ptr<llvm::orc::LLJIT>, std::function<void()>>
createWrappedKernel(std::string_view llvmIr, const std::string &kernelName,
                    void *args, std::uint64_t argsSize);
} // namespace cudaq

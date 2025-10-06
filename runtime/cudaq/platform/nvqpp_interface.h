/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ThunkInterface.h"
#include <cstdint>
#include <vector>

namespace cudaq {

/// Entry point for the auto-generated kernel execution path. TODO: Needs to be
/// tied to the quantum platform instance somehow. Note that the compiler cannot
/// provide that information.
extern "C" {
// Client-server (legacy) interface.
[[nodiscard]] KernelThunkResultType
altLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                std::uint64_t argsSize, std::uint64_t resultOffset);

// Streamlined interface for launching kernels. Argument synthesis and JIT
// compilation *must* happen on the local machine.
[[nodiscard]] KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs);

// Hybrid of the client-server and streamlined approaches. Letting JIT
// compilation happen either early or late and can handle return values from
// each kernel launch.
[[nodiscard]] KernelThunkResultType
hybridLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                   std::uint64_t argsSize, std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs);

//===----------------------------------------------------------------------===//
// Launch module entry points.
//
// In some environments (e.g., Python), the ModuleOp of the source can be
// provided immediately to be launched, unlike with statically compiled systems
// (C++). These entry points allow the managed runtime to provide the ModuleOp
// directly.
//===----------------------------------------------------------------------===//

// Client-server interface. The caller must provide an mlir::ModuleOp and the
// exact name of the entry point kernel function to be called, which is
// typically the .thunk unmarshalling function. Passing short names is
// considered incorrect.
[[nodiscard]] KernelThunkResultType
altLaunchModule(const char *exactEntryPointName, void *moduleOp, void *args,
                std::uint64_t argsSize, std::uint64_t resultOffset);

// Streamlined interface for launching kernels. Argument synthesis and JIT
// compilation *must* happen on the local machine. The caller must provide an
// mlir::ModuleOp and the exact name of the entry point kernel function to be
// called,
[[nodiscard]] KernelThunkResultType
streamlinedLaunchModule(const char *exactEntryPointName, void *moduleOp,
                        const std::vector<void *> &rawArgs);

// Hybrid of the client-server and streamlined approaches. Letting JIT
// compilation happen either early or late and can handle return values from
// each kernel launch. The caller must provide an mlir::ModuleOp and the exact
// name of the entry point kernel function to be called,
[[nodiscard]] KernelThunkResultType
hybridLaunchModule(const char *exactEntryPointName, void *moduleOp, void *args,
                   std::uint64_t argsSize, std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs);
} // extern "C"
} // namespace cudaq

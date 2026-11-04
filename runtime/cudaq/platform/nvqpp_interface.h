/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "common/ThunkInterface.h"
#include <vector>

namespace mlir {
class ModuleOp;
class Type;
} // namespace mlir

namespace cudaq {
struct LaunchContext;

/// Entry point for the auto-generated kernel execution path. TODO: Needs to be
/// tied to the quantum platform instance somehow. Note that the compiler cannot
/// provide that information.
///
/// `launchCtx` is the launch context that the kernel launch stub claimed with
/// `__cudaq_claimLaunchContext`. When it is null, no algorithm handed one over,
/// and the kernel launches with the platform's default launch context.
extern "C" {
// Claim the launch context that the algorithm calling the kernel handed over.
// Returns null if there is none. See `detail::claimLaunchContext`.
[[nodiscard]] LaunchContext *__cudaq_claimLaunchContext();

// Client-server (legacy) interface.
[[nodiscard]] KernelThunkResultType
altLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                std::uint64_t argsSize, std::uint64_t resultOffset,
                const LaunchContext *launchCtx);

// Streamlined interface for launching kernels. Argument synthesis and JIT
// compilation *must* happen on the local machine.
[[nodiscard]] KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs,
                        const LaunchContext *launchCtx);

// Hybrid of the client-server and streamlined approaches. Letting JIT
// compilation happen either early or late and can handle return values from
// each kernel launch.
[[nodiscard]] KernelThunkResultType
hybridLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                   std::uint64_t argsSize, std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs,
                   const LaunchContext *launchCtx);
} // extern "C"

//===----------------------------------------------------------------------===//
// Launch module entry points.
//
// In some environments (e.g., Python), the ModuleOp of the source can be
// provided immediately to be launched, unlike with statically compiled systems
// (C++). These entry points allow the managed runtime to provide the ModuleOp
// directly.
//===----------------------------------------------------------------------===//

// Streamlined interface for launching kernels, with the launch context that
// the caller claimed with `detail::claimLaunchContext` (or the platform's
// default one).
[[nodiscard]] KernelThunkResultType
streamlinedLaunchModule(const CompiledModule &compiled,
                        const std::vector<void *> &rawArgs,
                        const LaunchContext &launchCtx);

} // namespace cudaq

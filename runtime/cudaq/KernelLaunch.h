/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// The runtime's kernel launch API.

#include "common/CompiledModule.h"
#include "common/InvocationContext.h"
#include "common/KernelArgs.h"
#include "common/ThunkInterface.h"

namespace cudaq {

/// Claim the invocation context that the algorithm calling the kernel pushed,
/// or, if there is none, the default invocation context of the active
/// platform.
///
/// Call it exactly once per kernel launch.
[[nodiscard]] InvocationContext claimInvocationContext();

/// Launch \p module with \p args and the specified compile and launch
/// configuration.
// TODO: Once JIT compilation is moved out, this should no longer take
// `CompileTarget` and `CompileOptions` arguments (and `CompiledModule` would
// replace `AnyModule`).
[[nodiscard]] KernelThunkResultType
launchKernel(const AnyModule &module, KernelArgs args,
             const CompileTarget &target, const CompileOptions &compileOptions,
             const LaunchOptions &launchOptions, RuntimeEndpoint &endpoint,
             CompileModuleFn compileModule);

/// Launch \p module with \p args, as resolved by \p ctx.
[[nodiscard]] inline KernelThunkResultType
launchKernel(const AnyModule &module, KernelArgs args,
             const InvocationContext &ctx) {
  return launchKernel(module, args, ctx.target, ctx.compileOptions,
                      ctx.launchOptions, *ctx.endpoint, ctx.compileModule);
}

} // namespace cudaq

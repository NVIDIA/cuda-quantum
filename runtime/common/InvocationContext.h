/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompileOptions.h"
#include "common/CompiledModule.h"
#include "common/KernelArgs.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/algorithms/policies.h"
#include <cstddef>
#include <optional>

namespace cudaq {

class ExecutionContext;
struct RuntimeEndpoint;

/// The options that the runtime needs to launch a compiled kernel.
struct LaunchOptions {
  /// The QPU to launch on.
  std::size_t qpuId = 0;

  /// The policy to launch under, or no policy for a direct kernel launch.
  std::optional<AnyPolicy> policy{};

  /// The result produced by the policy launch. Points to the result type
  /// associated with `policy`.
  void *result = nullptr;

  /// The execution context in which to launch the kernel.
  // TODO: Remove this once the execution context is constructed by the QPU.
  ExecutionContext *executionContext = nullptr;
};

/// Compile a source module for a policy launch. Empty when compilation is
/// disabled, in which case the source module is wrapped without compilation.
using CompileModuleFn = CompiledModule (*)(CompileTarget target,
                                           CompileOptions options,
                                           const SourceModule &src,
                                           KernelArgs args, bool isEntryPoint);

/// The data that an algorithm hands to the kernel it launches.
///
/// This is passed from caller to callee through a dedicated global slot (see
/// `pushInvocationContext` and `popInvocationContext` below).
struct InvocationContext {
  /// The target to compile the kernel for.
  CompileTarget target;

  /// The options to compile the kernel with, resolved from the launch policy
  /// and the runtime endpoint.
  CompileOptions compileOptions;

  /// The options to launch the compiled kernel with.
  LaunchOptions launchOptions;

  /// The runtime endpoint to launch on. Never null. It is owned by the
  /// platform and outlives the kernel launch.
  RuntimeEndpoint *endpoint = nullptr;

  /// Compiles the kernel before a policy launch.
  // TODO: This callback is currently required because `cudaq-mlir-runtime` is
  // not linked in `libcudaq`. Fix this by running JIT compilation before kernel
  // launch.
  CompileModuleFn compileModule = nullptr;
};

namespace detail {

/// A RAII handle on the pending invocation context.
///
/// The invocation context is set on construction and cleared again on
/// destruction, if it was not claimed by then.
class [[nodiscard]] LiveInvocationContext {
public:
  explicit LiveInvocationContext(InvocationContext ctx);
  ~LiveInvocationContext();

  LiveInvocationContext(const LiveInvocationContext &) = delete;
  LiveInvocationContext &operator=(const LiveInvocationContext &) = delete;
};
} // namespace detail

/// Hand \p ctx over to the next kernel launch on this thread.
///
/// The kernel launch claims it with `cudaq::popInvocationContext`.
///
/// Throws if an invocation context is already pending. Only one may be pending
/// at any time.
///
/// Returns a RAII object that clears the context if it was not claimed by the
/// time the object goes out of scope.
[[nodiscard]] detail::LiveInvocationContext
pushInvocationContext(InvocationContext ctx);

/// Claim the pending invocation context, if any.
///
/// This is non-`nullopt` if and only if the algorithm calling the kernel
/// pushed a context before the kernel launch. Use
/// `cudaq::claimInvocationContext` to fall back to the platform's default.
[[nodiscard]] std::optional<InvocationContext> popInvocationContext() noexcept;

} // namespace cudaq

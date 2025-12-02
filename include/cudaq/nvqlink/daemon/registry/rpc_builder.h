/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/daemon/registry/function_traits.h"
#include "cudaq/nvqlink/daemon/registry/function_wrapper.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaq::nvqlink {

// Forward declaration
class Daemon;

/// Fluent builder API for bulk RPC function registration.
///
/// Allows registering multiple functions with a clean, chainable syntax:
/// @example
/// ```cpp
/// RPCBuilder()
///     .add(NVQLINK_RPC_HANDLE(rpc::add))
///     .add(NVQLINK_RPC_HANDLE(rpc::multiply))
///     .add(NVQLINK_RPC_HANDLE(rpc::echo))
///     .register_all(*daemon);
/// ```
///
class RPCBuilder {
public:
  RPCBuilder() = default;

  /// Add a function to the builder.
  ///
  /// Use with NVQLINK_RPC_HANDLE macro for automatic name extraction:
  /// ```cpp
  /// builder.add(NVQLINK_RPC_HANDLE(namespace::function));
  /// ```
  ///
  /// @tparam F Function type (deduced)
  /// @param func Function to register
  /// @param name Fully-qualified function name
  /// @return Reference to this builder (for chaining)
  ///
  template <typename F>
  RPCBuilder &add(F &&func, std::string_view name) {
    using Traits = function_traits<std::remove_cvref_t<F>>;
    using Return = typename Traits::return_type;

    FunctionMetadata meta{.function_id = hash_name(name),
                          .name = std::string(name),
                          .type = FunctionType::CPU,
                          .max_result_size = serialized_size<Return>(),
                          .cpu_function = make_wrapper(std::forward<F>(func)),
                          .gpu_function = nullptr};

    entries_.push_back(std::move(meta));
    return *this;
  }

  /// Register all accumulated functions with a daemon.
  ///
  /// @param daemon Daemon instance to register functions with
  /// @throws std::runtime_error if hash collision is detected
  ///
  void register_all(Daemon &daemon) const;

  /// Get the number of functions in the builder.
  ///
  /// @return Number of functions accumulated
  ///
  std::size_t size() const { return entries_.size(); }

  /// Check if the builder is empty.
  ///
  /// @return true if no functions have been added
  ///
  bool empty() const { return entries_.empty(); }

private:
  std::vector<FunctionMetadata> entries_;
};

} // namespace cudaq::nvqlink

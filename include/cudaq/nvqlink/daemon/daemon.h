/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"
#include "cudaq/nvqlink/network/config.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatcher.h"
#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/daemon/registry/function_traits.h"
#include "cudaq/nvqlink/daemon/registry/function_wrapper.h"

#include <atomic>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace cudaq::nvqlink {

/// Main daemon class representing an RPC server instance.
/// Configuration is immutable after construction.
class Daemon {
public:
  explicit Daemon(const DaemonConfig &config, std::unique_ptr<Channel> channel);
  ~Daemon();

  // Delete copy/move to prevent configuration changes
  Daemon(const Daemon &) = delete;
  Daemon &operator=(const Daemon &) = delete;
  Daemon(Daemon &&) = delete;
  Daemon &operator=(Daemon &&) = delete;

  // Lifecycle management
  void start();
  void stop();
  bool is_running() const { return running_.load(); }

  // Function registration (must be done before start())
  void register_function(const FunctionMetadata &metadata);

  /// Register a normal C++ function as an RPC handler (new intuitive API).
  ///
  /// Use with NVQLINK_RPC_HANDLE macro for automatic name extraction:
  /// @example
  /// ```cpp
  /// daemon->register_function(NVQLINK_RPC_HANDLE(namespace::function));
  /// ```
  ///
  /// @tparam F Function type (deduced)
  /// @param func Function to register
  /// @param name Fully-qualified function name (used for ID generation)
  /// @throws std::runtime_error if daemon is running or hash collision detected
  ///
  template <typename F>
  void register_function(F &&func, std::string_view name) {
    using Traits = function_traits<std::remove_cvref_t<F>>;
    using Return = typename Traits::return_type;

    FunctionMetadata meta{
        .function_id = hash_name(name),
        .name = std::string(name),
        .type = FunctionType::CPU,
        .max_result_size = serialized_size<Return>(),
        .cpu_function = make_wrapper(std::forward<F>(func)),
        .gpu_function = nullptr};

    register_function(meta); // Calls existing method, checks collision
  }

  // Statistics
  struct Stats {
    uint64_t packets_received{0};
    uint64_t packets_sent{0};
    uint64_t errors{0};
  };
  Stats get_stats() const;

  // Access to backend (for advanced usage)
  Channel *channel() { return channel_.get(); }

private:
  void validate_config() const;
  void initialize_dispatcher();

  DaemonConfig config_;
  std::atomic<bool> running_{false};

  std::unique_ptr<Channel> channel_;
  std::unique_ptr<FunctionRegistry> function_registry_;
  std::unique_ptr<Dispatcher> dispatcher_;

  std::vector<std::thread> worker_threads_;
  Stats stats_;
};

} // namespace cudaq::nvqlink

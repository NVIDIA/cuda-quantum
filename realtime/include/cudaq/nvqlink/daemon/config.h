/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::nvqlink {

/// Datapath execution mode
enum class DatapathMode { CPU, GPU };

/// Compute resource configuration for daemon execution
struct ComputeConfig {
  // CPU mode
  std::vector<uint32_t> cpu_cores;

  // GPU mode
  std::optional<uint32_t> gpu_device_id;
  uint32_t gpu_blocks{0};
  uint32_t gpu_threads_per_block{0};
};

/// Daemon configuration (Layer 3 - RPC server)
/// Daemon-specific settings for function dispatch and execution
struct DaemonConfig {
  std::string id;
  DatapathMode datapath_mode;
  ComputeConfig compute;

  // Validation
  bool is_valid() const;
};

/// Builder for DaemonConfig
class DaemonConfigBuilder {
public:
  DaemonConfigBuilder &set_id(const std::string &id);
  DaemonConfigBuilder &set_datapath_mode(DatapathMode mode);
  DaemonConfigBuilder &set_cpu_cores(const std::vector<uint32_t> &cores);
  DaemonConfigBuilder &set_gpu_device(uint32_t device_id, uint32_t blocks,
                                      uint32_t threads);

  DaemonConfig build() const;

private:
  DaemonConfig config_;
};

} // namespace cudaq::nvqlink

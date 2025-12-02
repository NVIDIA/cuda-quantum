/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::nvqlink {

enum class DatapathMode { CPU, GPU };

/// Channel configuration (Layer 1 - Hardware abstraction)
/// Each channel is 1:1 with a queue
struct ChannelConfig {
  std::string nic_device; // PCIe address or device name
  uint32_t queue_id;      // Single queue (1:1 with channel)

  // Buffer pool settings
  size_t pool_size_bytes{64 * 1024 * 1024}; // 64MB default
  size_t buffer_size_bytes{2048};           // 2KB default
  size_t headroom_bytes{256};
  size_t tailroom_bytes{64};
};

struct ComputeConfig {
  // CPU mode
  std::vector<uint32_t> cpu_cores;

  // GPU mode
  std::optional<uint32_t> gpu_device_id;
  uint32_t gpu_blocks{0};
  uint32_t gpu_threads_per_block{0};
};

/// Memory pool configuration (helper for MemoryPool construction)
struct MemoryConfig {
  size_t pool_size_bytes{64 * 1024 * 1024};
  size_t buffer_size_bytes{2048};
  size_t headroom_bytes{256};
  size_t tailroom_bytes{64};
};

/// Daemon configuration (Layer 3 - RPC server)
/// Simplified: only daemon-specific settings
struct DaemonConfig {
  std::string id;
  DatapathMode datapath_mode;
  ComputeConfig compute;

  // Validation
  bool is_valid() const;
};

// Simplified builder for DaemonConfig only
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

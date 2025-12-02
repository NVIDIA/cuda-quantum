/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string>

namespace cudaq::nvqlink {

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

/// Memory pool configuration (helper for MemoryPool construction)
struct MemoryConfig {
  size_t pool_size_bytes{64 * 1024 * 1024};
  size_t buffer_size_bytes{2048};
  size_t headroom_bytes{256};
  size_t tailroom_bytes{64};
};

} // namespace cudaq::nvqlink

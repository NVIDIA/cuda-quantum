/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/roce/verbs_context.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <cstring>
#include <stdexcept>

namespace cudaq::nvqlink {

VerbsContext::VerbsContext(const std::string &device_name)
    : device_name_(device_name) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "VerbsContext::VerbsContext");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "VerbsContext: Initializing for device {}",
                   device_name_);

  open_device();
  create_protection_domain();

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "VerbsContext: Successfully initialized for device {}",
                   device_name_);
}

VerbsContext::~VerbsContext() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "VerbsContext::~VerbsContext");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "VerbsContext: Cleaning up device {}",
                   device_name_);

  // Deallocate Protection Domain
  if (pd_) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
  }

  // Close device context
  if (context_) {
    ibv_close_device(context_);
    context_ = nullptr;
  }
}

void VerbsContext::open_device() {
  // Get list of available InfiniBand devices
  int num_devices;
  struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0)
    throw std::runtime_error("No InfiniBand devices found");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "VerbsContext: Found {} InfiniBand device(s)", num_devices);

  // Find device by name (device_name_)
  struct ibv_device *device = nullptr;
  for (int i = 0; i < num_devices; i++) {
    const char *dev_name = ibv_get_device_name(dev_list[i]);
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "  [{}] {}", i, dev_name);

    if (device_name_ == dev_name) {
      device = dev_list[i];
      break;
    }
  }

  // If not found by name, try using index
  if (!device) {
    try {
      int dev_idx = std::stoi(device_name_);
      if (dev_idx >= 0 && dev_idx < num_devices) {
        device = dev_list[dev_idx];
        NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                         "VerbsContext: Using device index {}: {}", dev_idx,
                         ibv_get_device_name(device));
      }
    } catch (const std::invalid_argument &) {
      // Not a number, device not found
    }
  }

  if (!device) {
    ibv_free_device_list(dev_list);
    throw std::runtime_error("Device '" + device_name_ + "' not found");
  }

  // Open device context
  context_ = ibv_open_device(device);
  if (!context_) {
    ibv_free_device_list(dev_list);
    throw std::runtime_error("Failed to open device '" + device_name_ + "'");
  }

  // Free device list immediately after opening device - no longer needed
  ibv_free_device_list(dev_list);

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "VerbsContext: Opened device {}",
                   ibv_get_device_name(context_->device));
}

void VerbsContext::create_protection_domain() {
  pd_ = ibv_alloc_pd(context_);
  if (!pd_)
    throw std::runtime_error("Failed to allocate Protection Domain");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "VerbsContext: Created Protection Domain");
}

} // namespace cudaq::nvqlink

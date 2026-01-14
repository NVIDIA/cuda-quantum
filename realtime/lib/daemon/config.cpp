/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/config.h"

using namespace cudaq::nvqlink;

bool DaemonConfig::is_valid() const {
  if (id.empty())
    return false;
  // Validation for datapath-specific config
  if (datapath_mode == DatapathMode::CPU && compute.cpu_cores.empty())
    return false;
  if (datapath_mode == DatapathMode::GPU && !compute.gpu_device_id.has_value())
    return false;
  return true;
}

DaemonConfigBuilder &DaemonConfigBuilder::set_id(const std::string &id) {
  config_.id = id;
  return *this;
}

DaemonConfigBuilder &DaemonConfigBuilder::set_datapath_mode(DatapathMode mode) {
  config_.datapath_mode = mode;
  return *this;
}

DaemonConfigBuilder &
DaemonConfigBuilder::set_cpu_cores(const std::vector<std::uint32_t> &cores) {
  config_.compute.cpu_cores = cores;
  return *this;
}

DaemonConfigBuilder &DaemonConfigBuilder::set_gpu_device(
    std::uint32_t device_id, std::uint32_t blocks, std::uint32_t threads) {
  config_.compute.gpu_device_id = device_id;
  config_.compute.gpu_blocks = blocks;
  config_.compute.gpu_threads_per_block = threads;
  return *this;
}

DaemonConfig DaemonConfigBuilder::build() const { return config_; }

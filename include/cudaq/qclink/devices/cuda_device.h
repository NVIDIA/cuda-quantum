/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qclink/device.h"

#include <map>

namespace cudaq::qclink {
class cuda_device : public device_mixin<explicit_data_marshalling_trait,
                                        device_callback_trait> {
  /// CUDA device ordinal this channel is associated with.
  int cudaDevice = 0;

  /// Paths to CUDA fatbin (module) files to be loaded.
  std::vector<std::string> fatbinLocations;

  /// Map of device pointer handles to raw device memory pointers.
  std::map<std::size_t, void *> local_memory_pool;

public:
  using device_mixin::device_mixin;
  cuda_device(std::size_t cudaDevId,
              const std::vector<std::string> &fatbinFiles)
      : device_mixin(), cudaDevice(cudaDevId), fatbinLocations(fatbinFiles) {}

  using explicit_data_marshalling_trait::free;
  using explicit_data_marshalling_trait::malloc;
  using device_callback_trait::launch_callback;

  void connect() override;
  void disconnect() override;
  void *resolve_pointer(device_ptr &devPtr) override;
  device_ptr malloc(std::size_t size) override;
  void free(device_ptr &d) override;
  void send(device_ptr &dest, const void *src) override;
  void recv(void *dest, const device_ptr &src) override;
  void launch_callback(const std::string &funcName, device_ptr &result,
                       const std::vector<device_ptr> &args) override;
};
} // namespace cudaq::qclink

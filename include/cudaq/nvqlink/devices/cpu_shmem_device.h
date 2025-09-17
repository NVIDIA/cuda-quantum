/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/nvqlink/device.h"

namespace cudaq::nvqlink {

class cpu_shmem_device
    : public device_mixin<explicit_data_marshalling_trait<cpu_shmem_device>,
                          device_callback_trait<cpu_shmem_device>> {
protected:
  std::vector<void *> handles;
  std::unordered_map<std::string, std::pair<void *, device_function>>
      whatIsThisCalled;

public:
  using device_mixin::device_mixin;
  void connect() override;
  void disconnect() override;
  void *resolve_pointer(device_ptr &devPtr);
  device_ptr malloc(std::size_t size);
  void free(device_ptr &d);
  void send(device_ptr &dest, const void *src);
  void recv(void *dest, const device_ptr &src);
  void launch_callback(const std::string &funcName, device_ptr &result,
                       const std::vector<device_ptr> &args);
};

} // namespace cudaq::nvqlink

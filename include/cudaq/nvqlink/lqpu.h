/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "device.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace cudaq::nvqlink {

class lqpu {

private:
  std::vector<std::unique_ptr<device>> m_devices;
public:
  lqpu(std::vector<std::unique_ptr<device>> &&devices)
      : m_devices(std::move(devices)) {
    for (auto &dev : m_devices)
      dev->connect();
  }

  ~lqpu() {
    for (auto &dev : m_devices)
      dev->disconnect();
  }

  std::size_t get_num_devices() const { return m_devices.size(); }
  std::size_t get_num_qcs_devices() const {
    std::size_t num = 0;
    for (auto &dev : m_devices)
      if (dev->isa<qcs_trait>())
        num++;
    return num;
  }

  // Provides access, throws on bad index for safety.
  device &get_device(std::size_t idx) {
    if (idx >= m_devices.size())
      throw std::out_of_range("Invalid device index");
    return *m_devices[idx];
  }
  const device &get_device(std::size_t idx) const {
    if (idx >= m_devices.size())
      throw std::out_of_range("Invalid device index");
    return *m_devices[idx];
  }

  const auto &get_devices() const { return m_devices; }

  std::vector<qcs_trait *> get_quantum_control_devices() const {
    std::vector<qcs_trait *> devs;
    for (auto &dev : m_devices)
      if (dev->isa<qcs_trait>())
        devs.push_back(dev->as<qcs_trait>());
    return devs;
  }
  
};

} // namespace cudaq::nvqlink

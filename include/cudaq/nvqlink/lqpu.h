/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "devices/all_devices.h"

#include <stdexcept>
#include <vector>

namespace cudaq::nvqlink {

class lqpu {

private:
  std::vector<any_device> m_devices;
  std::vector<any_device> m_qcs_devices;

public:
  lqpu(const std::vector<any_device> &devices) {
    for (auto &dev : const_cast<std::vector<any_device> &>(devices))
      std::visit(
          [&](auto &&x) mutable {
            using DeviceType = decltype(x);
            x.connect();
            if constexpr (has_qcs_trait_v<DeviceType>) {
              m_qcs_devices.push_back(x);
            } else {
              m_devices.push_back(x);
            }
          },
          dev);
  }

  ~lqpu() {
    for (auto &dev : m_qcs_devices)
      std::visit([](auto &&x) { x.disconnect(); }, dev);
    for (auto &dev : m_devices)
      std::visit([](auto &&d) { d.disconnect(); }, dev);
  }

  std::size_t get_num_devices() const { return m_devices.size(); }
  std::size_t get_num_qcs_devices() const { return m_qcs_devices.size(); }

  // Provides access, throws on bad index for safety.
  any_device &get_device(std::size_t idx) {
    if (idx >= m_devices.size())
      throw std::out_of_range("Invalid device index");
    return m_devices[idx];
  }
  const any_device &get_device(std::size_t idx) const {
    if (idx >= m_devices.size())
      throw std::out_of_range("Invalid device index");
    return m_devices[idx];
  }

  std::vector<any_device> get_quantum_control_devices() const {
    return m_qcs_devices;
  }
};

} // namespace cudaq::nvqlink

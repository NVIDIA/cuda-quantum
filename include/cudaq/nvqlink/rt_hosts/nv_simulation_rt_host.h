/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/nvqlink/rt_host.h"

namespace cudaq::nvqlink {

class nv_simulation_rt_host : public rt_host<nv_simulation_rt_host> {
public:
  using rt_host::rt_host;

  std::unique_ptr<compiler> get_compiler() const {
    return compiler::get("cudaq");
  }

  void trigger_execution(device_ptr &result,
                         const std::vector<device_ptr> &args) {
    for (auto &[i, dev] : m_quantum_devices)
      std::visit(
          [&](auto &&d) {
            using DeviceType = decltype(d);
            if constexpr (has_qcs_trait_v<DeviceType>) {
              d.trigger(result, args);
            } else {
              throw std::runtime_error("invalid device provided to "
                                       "nv_simulation_host trigger execution.");
            }
          },
          dev);
  }
};

} // namespace cudaq::nvqlink

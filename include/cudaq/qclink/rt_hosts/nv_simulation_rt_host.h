/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qclink/rt_host.h"

namespace cudaq::qclink {

class nv_simulation_rt_host : public rt_host {

protected:
  std::unique_ptr<compiler> get_compiler() override {
    return compiler::get("cudaq");
  }

  void trigger_execution(device_ptr &result,
                         const std::vector<device_ptr> &args) override {
    for (auto &[i, dev] : m_quantum_devices)
      dev->trigger(result, args);
  }

public:
  using rt_host::rt_host;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      nv_simulation_rt_host, "nv_simulation_rt_host",
      static std::unique_ptr<rt_host> create(lqpu &cfg) {
        return std::make_unique<nv_simulation_rt_host>(cfg);
      })
};

CUDAQ_REGISTER_TYPE(nv_simulation_rt_host)

} // namespace cudaq::qclink

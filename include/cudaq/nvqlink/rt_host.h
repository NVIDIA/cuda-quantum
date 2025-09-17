/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "compiler.h"
#include "devices/all_devices.h"
#include "lqpu.h"

#include <unordered_map>

namespace cudaq::nvqlink {

template <typename Derived>
class rt_host {
protected:
  lqpu &m_qpu;

  // Useful mapping of qdevice ids to the device type itself
  std::unordered_map<std::size_t, any_device> m_quantum_devices;

public:
  rt_host(lqpu &cfg) : m_qpu(cfg) {
    for (auto &qd : m_qpu.get_quantum_control_devices())
      m_quantum_devices.insert(
          {std::visit([](auto &&q) { return q.get_id(); }, qd), qd});
  }

  virtual ~rt_host() = default;

  std::unique_ptr<compiled_kernel> compile(const std::string &code,
                                           const std::string &kernel_name) {
    auto compiler = static_cast<const Derived *>(this)->get_compiler();
    auto qc_devices = m_qpu.get_quantum_control_devices();
    return compiler->compile(code, kernel_name, qc_devices.size());
  }

  void execute(compiled_kernel &kernel, const std::vector<device_ptr> &args,
               device_ptr &result) {
    for (const auto &prog : kernel.get_programs()) {
      auto it = m_quantum_devices.find(prog.qdevice_id);
      if (it == m_quantum_devices.end())
        throw std::runtime_error("Target quantum device with ID " +
                                 std::to_string(prog.qdevice_id) +
                                 " not found.");

      // Launch the program on the target device.
      std::visit(
          [&](auto &&d) {
            using DeviceType = decltype(d);
            if constexpr (has_qcs_trait_v<DeviceType>) {
              d.upload_program(prog.binary);
            } else {
              throw std::runtime_error("not a qcs device.");
            }
          },
          it->second);
    }

    static_cast<Derived *>(this)->trigger_execution(result, args);
  }
};

} // namespace cudaq::nvqlink

#include "rt_hosts/all_rt_hosts.h"

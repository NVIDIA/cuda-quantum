/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "compiler.h"
#include "device.h"
#include "lqpu.h"

#include "utils/extension_point.h"

#include <unordered_map>

namespace cudaq::qclink {

class rt_host : public cudaqx::extension_point<rt_host, lqpu&> {
protected:
  lqpu &m_qpu;

  // Useful mapping of qdevice ids to the device type itself
  std::unordered_map<std::size_t, qcs_trait *> m_quantum_devices;

  virtual void trigger_execution(device_ptr &res,
                                 const std::vector<device_ptr> &args) = 0;
  virtual std::unique_ptr<compiler> get_compiler() = 0;

public:
  rt_host(lqpu &cfg) : m_qpu(cfg) {
    for (auto &qd : m_qpu.get_devices())
      if (qd->isa<qcs_trait>())
        m_quantum_devices.insert({qd->get_id(), qd->as<qcs_trait>()});
  }

  virtual ~rt_host() = default;

  std::unique_ptr<compiled_kernel> compile(const std::string &code,
                                           const std::string &kernel_name) {
    auto compiler = get_compiler();
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
      it->second->upload_program(prog.binary);
    }

    trigger_execution(result, args);
  }
};

} // namespace cudaq::qclink

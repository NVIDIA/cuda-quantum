/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq_toolchain.h"
#include "cudaq/nvqlink/compiler.h"
#include "cudaq/nvqlink/device.h"

namespace cudaq::nvqlink {

// Subclasses need to fill the passes vector at
// construction, can also provide the -load pluginlib.so here
class cudaq_compiler : public compiler {
protected:
  std::vector<std::string> passes;

public:
  cudaq_compiler() {}
  bool understands_code(const std::string &code) const override;

  std::unique_ptr<compiled_kernel>
  compile(const std::string &code, const std::string &kernel_name,
          std::size_t num_qcs_devices) override;
};

} // namespace cudaq::nvqlink

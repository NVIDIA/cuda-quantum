/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "fomac/FoMaC.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {

using FoMaCDevice = typename decltype(std::declval<fomac::Session &>()
                                          .getDevices())::value_type;

struct QDMIDevice {
  explicit QDMIDevice(FoMaCDevice device) : device(std::move(device)) {}

  FoMaCDevice device;
  QDMI_Program_Format programFormat = QDMI_PROGRAM_FORMAT_QASM2;
  std::string name;
  std::size_t qubitCount = 0;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>> connectivity;
  bool isSimulator = false;
  bool isRemote = true;
};

} // namespace cudaq

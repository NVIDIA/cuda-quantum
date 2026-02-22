/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ServerHelper.h"
#ifdef CUDAQ_PASQAL_QRMI_ENABLED
#include "qrmi.h"
#endif

namespace cudaq {

class PasqalQrmiServerHelper {
public:
#ifdef CUDAQ_PASQAL_QRMI_ENABLED
  struct ModeConfig {
    std::string backendName;
  };

  static ModeConfig resolveConfig();
  static std::vector<ServerMessage>
  createJobs(std::vector<KernelExecution> &circuitCodes, std::size_t shots,
             const std::string &backendName);
  static ExecutionResult
  parseCountsFromTaskResult(const std::string &taskResultJson);
#endif
};

} // namespace cudaq

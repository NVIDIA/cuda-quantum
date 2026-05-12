/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteSimulatorQPU.h"

namespace cudaq {

/// Remote QPU: delegating the execution to a remotely-hosted server, which can
/// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public BaseRemoteSimulatorQPU {
public:
  RemoteSimulatorQPU();
  RemoteSimulatorQPU(RemoteSimulatorQPU &&) = delete;
  ~RemoteSimulatorQPU() override;
};

} // namespace cudaq

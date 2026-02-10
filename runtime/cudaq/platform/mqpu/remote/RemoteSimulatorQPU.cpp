/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRemoteSimulatorQPU.h"
#include "common/RuntimeMLIR.h"

using namespace mlir;

namespace {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {

public:
  RemoteSimulatorQPU() : BaseRemoteSimulatorQPU() {
    m_mlirContext = cudaq::getOwningMLIRContext();
  }

  RemoteSimulatorQPU(RemoteSimulatorQPU &&) = delete;
  virtual ~RemoteSimulatorQPU() = default;
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteSimulatorQPU, RemoteSimulatorQPU)

/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteSimulatorQPU.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"

cudaq::RemoteSimulatorQPU::RemoteSimulatorQPU() : BaseRemoteSimulatorQPU() {
  m_mlirContext = cudaq_internal::compiler::getOwningMLIRContext();
}

cudaq::RemoteSimulatorQPU::~RemoteSimulatorQPU() = default;

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::RemoteSimulatorQPU, RemoteSimulatorQPU)

/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "simulator_cutensornet.h"

namespace nvqir {
class SimulatorTensorNet : public SimulatorTensorNetBase {
public:
  SimulatorTensorNet() : SimulatorTensorNetBase() {
    // tensornet backend supports distributed tensor network contraction,
    // i.e., distributing tensor network contraction across multiple
    // GPUs/processes.
    //
    // Note: this requires CUTENSORNET_COMM_LIB as described in
    // the Getting Started section of the cuTensorNet library documentation
    // (Installation and Compilation).
    if (cudaq::mpi::is_initialized())
      initCuTensornetComm(m_cutnHandle);
  }
  // Nothing to do for state preparation
  virtual void prepareQubitTensorState() override {}
  virtual std::string name() const override { return "tensornet"; }
};
} // namespace nvqir

/// Register this Simulator class with NVQIR under name "tensornet"
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorTensorNet, tensornet)

/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "simulator_cutensornet.h"

namespace nvqir {
class SimulatorTensorNet : public SimulatorTensorNetBase {
public:
  // Nothing to do for state preparation
  virtual void prepareQubitTensorState() override {}
  virtual std::string name() const override { return "tensornet"; }
};
} // namespace nvqir

/// Register this Simulator class with NVQIR under name "tensornet"
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorTensorNet, tensornet)

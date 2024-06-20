/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/PluginUtils.h"
#include "nvqir/CircuitSimulator.h"

namespace nvqir {
extern CircuitSimulator *getCircuitSimulatorInternal();
}

namespace cudaq {

/// @brief Return the quantum circuit simulator for qubits.
nvqir::CircuitSimulator *get_simulator() {
  return nvqir::getCircuitSimulatorInternal();
}

} // namespace cudaq
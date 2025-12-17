/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ExecutionContext.h"
#include "common/PluginUtils.h"
#include "nvqir/CircuitSimulator.h"
#include <memory>
#include <optional>
#include <string_view>

namespace nvqir {
extern std::unique_ptr<CircuitSimulator>
createSimulator(std::optional<std::string_view> name = std::nullopt);
}

namespace cudaq {

/// @brief Return the quantum circuit simulator for qubits.
inline nvqir::CircuitSimulator *get_simulator() {
  auto ctx = getExecutionContext();
  if (!ctx || !ctx->simulationContext)
    return nullptr;

  return &ctx->simulationContext->getSimulator();
}

} // namespace cudaq

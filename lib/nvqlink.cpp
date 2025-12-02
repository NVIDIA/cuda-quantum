/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/nvqlink.h"

namespace cudaq::nvqlink {

void initialize(const LQPUConfig& cfg) {
  // TODO: Initialize based on new architecture
  // Stub implementation for Phase 1
}

void shutdown() {
  // TODO: Clean shutdown of all components
  // Stub implementation for Phase 1
}

} // namespace cudaq::nvqlink

extern "C" {

// Old dispatch function removed - will be replaced by Daemon's FunctionRegistry
// When needed, device_call will be routed through the Daemon

}

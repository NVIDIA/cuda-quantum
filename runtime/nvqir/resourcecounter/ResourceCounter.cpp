/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResourceCounter.h"

namespace nvqir {
// Should be alive for the whole runtime, so won't leak memory
thread_local ResourceCounter *resource_counter_simulator = nullptr;

ResourceCounter *getResourceCounterSimulator() {
  if (!resource_counter_simulator)
    resource_counter_simulator = new nvqir::ResourceCounter();

  return resource_counter_simulator;
}

void setChoiceFunction(std::function<bool()> choice) {
  getResourceCounterSimulator()->setChoiceFunction(choice);
}

cudaq::Resources *getResourceCounts() {
  getResourceCounterSimulator()->flushGateQueue();
  return getResourceCounterSimulator()->getResourceCounts();
}
} // namespace nvqir

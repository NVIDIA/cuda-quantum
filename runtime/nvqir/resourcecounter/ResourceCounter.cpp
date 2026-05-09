/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResourceCounter.h"
#include "cudaq/analysis/resource_counter.h"
#include <stdexcept>
#include <utility>

namespace nvqir {
// Per-thread singleton; lives for the duration of the thread.
thread_local ResourceCounter *resource_counter_simulator = nullptr;

ResourceCounter *getResourceCounterSimulator() {
  if (!resource_counter_simulator)
    resource_counter_simulator = new nvqir::ResourceCounter();

  return resource_counter_simulator;
}

} // namespace nvqir

namespace cudaq::analysis::resource_counter {

scope make_scope(std::function<bool()> choice) {
  auto *rc = nvqir::getResourceCounterSimulator();
  rc->setChoiceFunction(std::move(choice));
  return scope{
      "resource_counter",
      *rc,
      {.on_enter = nullptr, .on_exit = [](nvqir::CircuitSimulator &sim) {
         static_cast<nvqir::ResourceCounter &>(sim).setToZeroState();
       }}};
}

cudaq::Resources get_counts(scope &s) {
  auto &rc = static_cast<nvqir::ResourceCounter &>(s.simulator());
  rc.flushGateQueue();
  return cudaq::Resources(*rc.getResourceCounts());
}

void prepopulate(cudaq::Resources counts) {
  if (!scope::is_active())
    throw std::runtime_error("`cudaq::analysis::resource_counter::prepopulate`:"
                             " no analysis scope is active on this thread.");
  auto *rc = nvqir::getResourceCounterSimulator();
  rc->flushGateQueue();
  rc->setResourceCounts(std::move(counts));
}

} // namespace cudaq::analysis::resource_counter

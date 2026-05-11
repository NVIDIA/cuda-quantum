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
  // Install the choice function only after the scope has successfully claimed
  // the thread-local slot.
  return scope{
      "resource_counter",
      *rc,
      {.on_enter =
           [rc, choice = std::move(choice)](nvqir::CircuitSimulator &) mutable {
             rc->setChoiceFunction(std::move(choice));
           },
       .on_exit = [rc](nvqir::CircuitSimulator &) { rc->setToZeroState(); }}};
}

cudaq::Resources get_counts(scope &s) {
  auto *rc = nvqir::getResourceCounterSimulator();
  // Reject scopes that are not backed by the resource-counter singleton so
  // callers can't accidentally reinterpret other plugin simulator
  // as a "ResourceCounter".
  if (&s.simulator() != rc)
    throw std::runtime_error(
        "`cudaq::analysis::resource_counter::get_counts`: scope is not a "
        "resource-counter scope.");
  rc->flushGateQueue();
  return cudaq::Resources(*rc->getResourceCounts());
}

void prepopulate(cudaq::Resources counts) {
  auto *rc = nvqir::getResourceCounterSimulator();
  if (scope::active_simulator() != rc)
    throw std::runtime_error(
        "`cudaq::analysis::resource_counter::prepopulate`: no resource-counter"
        " scope is active on this thread.");
  rc->flushGateQueue();
  rc->setResourceCounts(std::move(counts));
}

} // namespace cudaq::analysis::resource_counter

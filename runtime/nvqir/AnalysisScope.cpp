/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "common/PluginUtils.h"
#include "cudaq/analysis/scope.h"
#include "cudaq/runtime/logger/logger.h"
#include <stdexcept>
#include <utility>

namespace nvqir {

// Thread-local simulator override slot consulted by the NVQIR resolver in
// `getCircuitSimulatorInternal()`. A non-null value preempts the normal
// sampling backend and routes all gate / measurement / qubit-allocation
// calls to the analysis simulator until the owning `scope` is destroyed.
//
// Single-slot by design: nested `cudaq::analysis::scope` instances on the
// same thread throw at construction (see `scope::scope`). LIFO nesting can
// be added later by promoting this to a vector without breaking the public
// API.
thread_local CircuitSimulator *activeAnalysisSimulator = nullptr;

} // namespace nvqir

namespace cudaq::analysis {

scope::scope(std::string name, nvqir::CircuitSimulator &sim, hooks h)
    : name_(std::move(name)), sim_(&sim), on_exit_(std::move(h.on_exit)) {
  if (nvqir::activeAnalysisSimulator)
    throw std::runtime_error(
        "`cudaq::analysis::scope`: a scope is already active on this thread "
        "(nested analysis scopes are not supported).");
  nvqir::activeAnalysisSimulator = sim_;
  if (h.on_enter) {
    try {
      h.on_enter(*sim_);
    } catch (...) {
      // Release the slot before propagating so the thread is not left in an
      // active-scope state if on_enter throws.
      nvqir::activeAnalysisSimulator = nullptr;
      throw;
    }
  }
}

scope scope::from_plugin(std::string name, std::string plugin_name, hooks h) {
  const auto symbol = std::string("getCircuitSimulator_") + plugin_name;
  auto *sim = cudaq::getUniquePluginInstance<nvqir::CircuitSimulator>(symbol);
  if (!sim)
    throw std::runtime_error("`cudaq::analysis::scope::from_plugin`: plugin '" +
                             plugin_name +
                             "' returned a null CircuitSimulator.");
  return scope{std::move(name), *sim, std::move(h)};
}

scope::~scope() noexcept {
  if (on_exit_) {
    try {
      on_exit_(*sim_);
    } catch (const std::exception &e) {
      // CUDAQ_ERROR throws; that is fatal in a noexcept destructor. Use the
      // logging-only `cudaq::error` deduction struct instead so we record
      // the failure without escalating it through stack unwinding.
      cudaq::error("`cudaq::analysis::scope` '{}' on_exit threw: {}", name_,
                   e.what());
    } catch (...) {
      cudaq::error("`cudaq::analysis::scope` '{}' on_exit threw a non-std "
                   "exception",
                   name_);
    }
  }
  nvqir::activeAnalysisSimulator = nullptr;
}

bool scope::is_active() noexcept {
  return nvqir::activeAnalysisSimulator != nullptr;
}

} // namespace cudaq::analysis

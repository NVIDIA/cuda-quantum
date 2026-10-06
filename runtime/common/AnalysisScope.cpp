/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "AnalysisScope.h"
#include "PluginUtils.h"
#include "nvqir/CircuitSimulator.h"
#include "cudaq/runtime/logger/logger.h"
#include <stdexcept>
#include <utility>

namespace {

/// Thread-local simulator override slot consulted by the NVQIR resolver in
/// `getCircuitSimulatorInternal()`. A non-null value preempts the normal
/// sampling backend and routes all gate / measurement / qubit-allocation
/// calls to the analysis simulator until the owning `AnalysisScope` is
/// destroyed.
///
/// Single-slot by design: nested `cudaq::AnalysisScope` instances on the
/// same thread throw at construction (see `AnalysisScope::AnalysisScope`).
/// LIFO nesting can be added later by promoting this to a vector without
/// breaking the public API.
thread_local nvqir::CircuitSimulator *activeAnalysisSimulator = nullptr;

} // namespace

cudaq::detail::AnalysisScope::AnalysisScope(std::string name,
                                            nvqir::CircuitSimulator &sim,
                                            hooks h)
    : name_(std::move(name)), sim_(&sim), on_exit_(std::move(h.on_exit)) {
  if (activeAnalysisSimulator)
    throw std::runtime_error(
        "`cudaq::AnalysisScope`: a scope is already active on this thread "
        "(nested analysis scopes are not supported).");
  activeAnalysisSimulator = sim_;
  if (h.on_enter) {
    try {
      h.on_enter(*sim_);
    } catch (...) {
      // Release the slot before propagating so the thread is not left in
      // an active-scope state if on_enter throws.
      activeAnalysisSimulator = nullptr;
      throw;
    }
  }
}

cudaq::detail::AnalysisScope
cudaq::detail::AnalysisScope::from_plugin(std::string name,
                                          std::string plugin_name, hooks h) {
  const auto symbol = std::string("getCircuitSimulator_") + plugin_name;
  auto *sim = cudaq::getUniquePluginInstance<nvqir::CircuitSimulator>(symbol);
  if (!sim)
    throw std::runtime_error("`cudaq::AnalysisScope::from_plugin`: plugin '" +
                             plugin_name +
                             "' returned a null CircuitSimulator.");
  return AnalysisScope{std::move(name), *sim, std::move(h)};
}

cudaq::detail::AnalysisScope::~AnalysisScope() noexcept {
  if (on_exit_) {
    try {
      on_exit_(*sim_);
    } catch (const std::exception &e) {
      // CUDAQ_ERROR throws; that is fatal in a noexcept destructor. Use the
      // logging-only `cudaq::error` deduction struct instead so we record
      // the failure without escalating it through stack unwinding.
      cudaq::error("`cudaq::AnalysisScope` '{}' on_exit threw: {}", name_,
                   e.what());
    } catch (...) {
      cudaq::error("`cudaq::AnalysisScope` '{}' on_exit threw a non-std "
                   "exception",
                   name_);
    }
  }
  activeAnalysisSimulator = nullptr;
}

bool cudaq::detail::AnalysisScope::is_active() noexcept {
  return activeAnalysisSimulator != nullptr;
}

nvqir::CircuitSimulator *
cudaq::detail::AnalysisScope::active_simulator() noexcept {
  return activeAnalysisSimulator;
}

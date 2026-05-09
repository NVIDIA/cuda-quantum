/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq::analysis {

/// @brief RAII override of the active circuit simulator for non-sampling
/// analyses (resource counting, detector error model generation, ...).
///
/// While a `scope` is alive on the current thread, the NVQIR resolver routes
/// all gate, measurement, and qubit-allocation calls to the simulator passed
/// to the constructor instead of the normal sampling backend. The slot is
/// thread-local: each thread can run its own analysis independently.
///
/// Construction takes ownership of the override; destruction releases it.
/// Nested scopes on the same thread throw, since downstream engines (e.g.
/// resource counting and DEM) are not designed to compose.
///
/// Engines (resource_counter, future DEM) provide their own factory
/// functions that return a fully configured `scope` — user code does not
/// instantiate `scope` directly.
class scope {
public:
  /// @brief Optional callbacks fired on entry and exit of the scope.
  ///
  /// `on_enter` runs after the slot has been claimed; useful for engines that
  /// need a known starting state. `on_exit` runs while the slot is still
  /// claimed, before the destructor releases it; the default resets the
  /// simulator to the |0...0> state to avoid leaking analysis residue into a
  /// subsequent run on the same thread.
  ///
  /// Both callbacks must not throw. Exceptions from `on_exit` are caught and
  /// logged; `on_enter` exceptions propagate out of the constructor and the
  /// slot is released before they do.
  struct hooks {
    std::function<void(nvqir::CircuitSimulator &)> on_enter;
    std::function<void(nvqir::CircuitSimulator &)> on_exit;
  };

  /// @brief Activate `sim` as the analysis simulator for the current thread.
  /// Throws `std::runtime_error` if a scope is already active on this thread.
  scope(std::string name, nvqir::CircuitSimulator &sim, hooks h = {});

  /// @brief Activate the simulator exposed by the named NVQIR plugin.
  ///
  /// Resolves `getCircuitSimulator_<plugin_name>` via dlsym in the current
  /// process's loaded symbol space and returns a `scope` that owns it. The
  /// plugin shared library must already be loaded (statically linked or
  /// dlopen'd by the caller).
  static scope from_plugin(std::string name, std::string plugin_name,
                           hooks h = {});

  ~scope() noexcept;

  scope(const scope &) = delete;
  scope &operator=(const scope &) = delete;
  scope(scope &&) = delete;
  scope &operator=(scope &&) = delete;

  /// @brief The underlying simulator. Stable for the lifetime of the scope.
  nvqir::CircuitSimulator &simulator() const noexcept { return *sim_; }

  /// @brief Identifier supplied at construction. Used for logs and diagnostics.
  std::string_view name() const noexcept { return name_; }

  /// @brief True iff a `scope` is currently active on the calling thread.
  static bool is_active() noexcept;

private:
  std::string name_;
  nvqir::CircuitSimulator *sim_;
  std::function<void(nvqir::CircuitSimulator &)> on_exit_;
};

} // namespace cudaq::analysis

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/QuditIdTracker.h"
#include "cudaq/spin_op.h"
#include <cassert>
#include <complex>
#include <cstddef>
#include <deque>
#include <span>
#include <string_view>
#include <vector>

namespace cudaq {
class ExecutionContext;
using SpinMeasureResult = std::pair<double, sample_result>;

// A QuditInfo, to the ExecutionManager, is a type encoding
// the number of levels and the id of the qudit.
struct QuditInfo {
  std::size_t levels = 0;
  std::size_t id = 0;
  QuditInfo(const std::size_t &_levels, const std::size_t &_id)
      : levels(_levels), id(_id) {}
  bool operator==(const QuditInfo &other) {
    return levels == other.levels && id == other.id;
  }
};

bool __nvqpp__MeasureResultBoolConversion(int);

#ifdef CUDAQ_LIBRARY_MODE

/// @brief In library mode, we model the return type of
/// a qubit measurement result via the measure_result type.
/// This allows us to keep track of when the result is
/// implicitly casted to a bool (likely in the case of
/// conditional feedback), and affect the simulation accordingly
class measure_result {
private:
  /// @brief The intrinsic measurement result
  int result = 0;

  /// @brief Unique integer for measure result identification
  std::size_t uniqueId = 0;

public:
  measure_result(int res, std::size_t id) : result(res), uniqueId(id) {}
  measure_result(int res) : result(res) {}

  operator int() { return result; }
  operator bool() { return __nvqpp__MeasureResultBoolConversion(result); }
};
#else
/// @brief When compiling with MLIR, we default to a boolean
using measure_result = bool;
#endif

/// The ExecutionManager provides a base class describing a
/// concrete sub-system for allocating qudits and executing quantum
/// instructions on those qudits. This type is templated on the concrete
/// qudit type (qubit, qmode, etc). It exposes an API for getting an
/// available qudit id, returning that id, setting and resetting the
/// current execution context, and applying specific quantum instructions.
class ExecutionManager {
protected:
  /// Available qudit indices
  std::deque<std::size_t> availableIndices;

  /// Total qudits available
  std::size_t totalQudits;

  /// @brief Utility type tracking qudit unique
  /// identifiers as they are allocated and deallocated.
  QuditIdTracker tracker;

  /// Internal - return the next qudit index
  std::size_t getNextIndex() { return tracker.getNextIndex(); }

  /// Internal - At qudit deallocation, return the qudit index
  void returnIndex(std::size_t idx) { tracker.returnIndex(idx); }

public:
  ExecutionManager() = default;

  /// Return the next available qudit index
  virtual std::size_t getAvailableIndex(std::size_t quditLevels = 2) = 0;

  /// QuditInfo has been deallocated, return the qudit / id to the pool of
  /// qudits.
  virtual void returnQudit(const QuditInfo &q) = 0;

  /// Checker for qudits that were not deallocated
  bool memoryLeaked() { return !tracker.allDeallocated(); }

  /// Provide an ExecutionContext for the current cudaq kernel
  virtual void setExecutionContext(cudaq::ExecutionContext *ctx) = 0;

  /// Reset the execution context
  virtual void resetExecutionContext() = 0;

  /// Apply the quantum instruction with the given name, on the provided
  /// target qudits. Supports input of control qudits and rotational parameters.
  /// Can also optionally take a spin_op as input to affect a general
  /// Pauli rotation.
  virtual void apply(const std::string_view gateName,
                     const std::vector<double> &params,
                     const std::vector<QuditInfo> &controls,
                     const std::vector<QuditInfo> &targets,
                     bool isAdjoint = false, const spin_op op = spin_op()) = 0;

  /// Reset the qubit to the |0> state
  virtual void reset(const QuditInfo &target) = 0;

  /// Begin an region of code where all operations will be adjoint-ed
  virtual void startAdjointRegion() = 0;
  /// End the adjoint region
  virtual void endAdjointRegion() = 0;

  /// Start a region of code where all operations will be
  /// controlled on the given qudits.
  virtual void
  startCtrlRegion(const std::vector<std::size_t> &control_qubits) = 0;
  /// End the control region
  virtual void endCtrlRegion(std::size_t n_controls) = 0;

  /// Measure the qudit and return the observed state (0,1,2,3,...)
  /// e.g. for qubits, this can return 0 or 1;
  virtual int measure(const QuditInfo &target) = 0;

  /// Measure the current state in the given pauli basis, return
  /// the expectation value <term>.
  virtual SpinMeasureResult measure(cudaq::spin_op &op) = 0;

  /// Synchronize - run all queue-ed instructions
  virtual void synchronize() = 0;
  virtual ~ExecutionManager() = default;
};

ExecutionManager *getExecutionManager();
} // namespace cudaq

// The following macro is to be used by ExecutionManager subclass
// developers. It will define the global thread_local execution manager
// pointer instance, and define the factory function for clients to
// get reference to the execution manager
#define CUDAQ_REGISTER_EXECUTION_MANAGER(Manager)                              \
  namespace cudaq {                                                            \
  ExecutionManager *getExecutionManager() {                                    \
    thread_local static std::unique_ptr<ExecutionManager> qis_manager =        \
        std::make_unique<Manager>();                                           \
    return qis_manager.get();                                                  \
  }                                                                            \
  }

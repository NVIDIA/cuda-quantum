/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Forward declaration only. Consumers that need to dereference the returned
// pointer include `<stim.h>` directly.
namespace stim {
struct Circuit;
} // namespace stim

namespace nvqir {

/// @brief Capability interface for the backend simulators that build up a
/// Stim-format recorded circuit during execution.
struct RecordedCircuit {
  virtual ~RecordedCircuit() = default;

  /// @brief Pointer to the simulator's accumulated `stim::Circuit`.
  virtual const stim::Circuit *circuit() const = 0;

  /// @brief Move the accumulated `stim::Circuit` out of the simulator,
  /// leaving the simulator with an empty recorded circuit.
  virtual stim::Circuit release() = 0;

  /// @brief Drop the accumulated circuit AND any per-execution simulator
  /// state so the next analysis run starts from a clean slate. Heavier
  /// than `release()`: also deallocates the underlying tableau / sample
  /// state when the backend has it.
  virtual void reset() = 0;
};

} // namespace nvqir

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Resources.h"
#include "nvqir/AnalysisScope.h"
#include <functional>

namespace nvqir::resource_counter {

/// @brief Activate the resource-counter analysis on the current thread.
///
/// Returns an `AnalysisScope` that, while alive, routes gate/measurement
/// traffic to the resource-counter simulator. `choice` is invoked for every
/// measurement to deterministically pick which branch to follow when the
/// kernel contains mid-circuit measurement-conditional logic.
///
/// Throws `std::runtime_error` if an analysis scope is already active on the
/// current thread.
AnalysisScope make_scope(std::function<bool()> choice);

/// @brief Snapshot of the resource counts accumulated so far.
///
/// Must be called while `s` is the active resource-counter scope. The result
/// is a value-typed copy; the underlying simulator state continues to evolve
/// as more gates are dispatched.
cudaq::Resources get_counts(AnalysisScope &s);

/// @brief Pre-populate the resource-counter simulator with counts harvested
/// from an MLIR-level analysis pass (`countResourcesFromIR`).
///
/// Used by the JIT path to short-circuit gate-by-gate counting once the
/// optimizer has already produced exact figures. Throws if no scope is
/// currently active — pre-population without an active scope would silently
/// leak counts into a future scope on the same thread.
void prepopulate(cudaq::Resources counts);

} // namespace nvqir::resource_counter

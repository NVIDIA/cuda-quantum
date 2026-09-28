/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// This is a local helper for the ExpPauli identity-Pauli (global phase)
// decomposition in DecompositionPatterns.cpp and its alias checks in
// PhaseUtilities.h. It is deliberately kept out of the Quake dialect's public
// headers: it is a special-case planning helper for that one rewrite, not a
// general dialect utility, and is more likely to need to change alongside
// that rewrite than to grow new, unrelated callers.

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <cstddef>
#include <optional>

namespace cudaq::quake {

/// A statically selectable scalar qubit represented by a top-level Quake
/// target. A vector target records the element that must be extracted; a
/// scalar reference or wire has no element index.
struct StaticQubitTarget {
  mlir::Value source;
  std::size_t sourceIndex;
  std::optional<std::size_t> elementIndex;
};

/// Plan a statically selectable scalar target without creating IR.
std::optional<StaticQubitTarget> planStaticQubitTarget(mlir::Value target,
                                                       std::size_t sourceIndex);

/// Plan the last scalar target accepted by \p predicate without creating IR.
template <typename Predicate>
std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets, Predicate predicate) {
  for (std::size_t i = targets.size(); i != 0; --i) {
    auto finalTarget = planStaticQubitTarget(targets[i - 1], i - 1);
    if (!finalTarget)
      continue;
    if (!finalTarget->elementIndex) {
      if (predicate(*finalTarget))
        return finalTarget;
      continue;
    }

    for (std::size_t element = *finalTarget->elementIndex + 1; element != 0;
         --element) {
      StaticQubitTarget candidate{finalTarget->source, finalTarget->sourceIndex,
                                  element - 1};
      if (predicate(candidate))
        return candidate;
    }
  }
  return std::nullopt;
}

/// Plan a deterministic final scalar target without creating IR.
std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets);

/// Materialize a target selected by findLastStaticQubitTarget.
mlir::Value materializeStaticQubitTarget(mlir::OpBuilder &builder,
                                         mlir::Location location,
                                         const StaticQubitTarget &target);

} // namespace cudaq::quake

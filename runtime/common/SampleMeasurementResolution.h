/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include <optional>
#include <stdexcept>

namespace cudaq::details {

inline bool
requestedNonExplicitMeasurements(std::optional<bool> requestedExplicit) {
  return requestedExplicit && !*requestedExplicit;
}

inline void
rejectUnsupportedSampleExplicitFalse(std::optional<bool> requestedExplicit,
                                     const ExecutionContext &ctx) {
  if (requestedNonExplicitMeasurements(requestedExplicit) &&
      ctx.explicitMeasurements)
    throw std::runtime_error(
        "This kernel requires explicit measurement result semantics for "
        "sampling, but `explicit_measurements=false` was requested.");
}

/// Resolve the execution mode for sampling after kernel measurement semantics
/// are known, or as much as we can know before launching the kernel.
///
/// `requiresExplicit` is the kernel requirement computed from analysis:
/// - std::nullopt: metadata is not available yet (for example, deprecated
///   library-mode paths or kernels whose IR is produced during launch).
/// - false: legacy/non-explicit sampling preserves the kernel result.
/// - true: sampling must preserve measurement execution order, duplicate
///   measurements, mixed bases, etc.
///
/// `requestedExplicit` is only the user request. A user-provided `false` asks
/// for legacy non-explicit sampling, but this function does not throw for that
/// conflict directly. Instead, it sets `ctx.explicitMeasurements` from the
/// kernel requirement and lets `rejectUnsupportedSampleExplicitFalse` report
/// the user-facing error once a concrete requirement is known.
///
/// When the kernel requirement is unknown, default/explicit-true requests use
/// the target's explicit-measurement capability as a conservative execution
/// mode. This preserves measurement-order behavior for paths without metadata,
/// while explicit-false requests still run in legacy mode unless later compiler
/// analysis updates the context and triggers rejection.
inline void resolveSampleExplicitMeasurements(
    ExecutionContext &ctx, std::optional<bool> requiresExplicit,
    bool targetSupportsExplicit,
    std::optional<bool> requestedExplicit = std::nullopt) {
  if (ctx.name != "sample")
    return;

  if (!requiresExplicit) {
    ctx.explicitMeasurements =
        requestedNonExplicitMeasurements(requestedExplicit)
            ? false
            : targetSupportsExplicit;
    return;
  }

  if (!*requiresExplicit) {
    ctx.explicitMeasurements = false;
    return;
  }

  if (!targetSupportsExplicit)
    throw std::runtime_error(
        "This kernel requires explicit measurement result semantics for "
        "sampling, but explicit measurements are not supported on this "
        "target.");

  ctx.explicitMeasurements = true;
}

} // namespace cudaq::details

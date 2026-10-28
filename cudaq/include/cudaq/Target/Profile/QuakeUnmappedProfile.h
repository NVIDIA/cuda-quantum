/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cudaq::config {

/// What the optimizer of a `quake-unmapped` stage minimizes.
enum class CostModel {
  /// Minimize T gates and non-Clifford rotations, which dominate the cost on
  /// fault-tolerant hardware.
  t_count,
};

/// IR profile for Quake before qubit mapping. Qubits come from allocations
/// rather than a device's wire set, and any two may interact. Wires and
/// references are both allowed.
///
/// Use it for a target that needs a fixed gate set but not a placement of
/// qubits on a device, such as a fault-tolerant target that counts logical
/// resources or hardware where any two qubits can interact. With a cost model,
/// the stage optimizes for that cost, and when the basis has no rotation gate
/// it synthesizes each rotation within the tolerance. Without a cost model,
/// the stage only converts to the basis. For example, a Clifford+T target
/// lists:
///
///   ir-output-profile:
///     name: quake-unmapped
///     options:
///       basis-gates: [h, s, t, x, y, z, swap, "x(1)", "y(1)", "z(1)"]
///       approximation-tolerance: 1e-10
///       cost-model: t-count
struct QuakeUnmappedProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "quake-unmapped";

  /// Gates the IR is restricted to, using the basis-conversion grammar. An
  /// empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Operator-norm tolerance for erasing, snapping, and synthesizing
  /// rotations. Zero accepts only exact rewrites, so synthesis needs a positive
  /// tolerance.
  double approximationTolerance = 0;

  /// Without a cost model, the stage runs no optimization.
  std::optional<CostModel> costModel = std::nullopt;

  bool operator==(const QuakeUnmappedProfile &other) const = default;
};

} // namespace cudaq::config

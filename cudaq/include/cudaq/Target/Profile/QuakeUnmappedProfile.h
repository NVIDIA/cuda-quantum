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
struct QuakeUnmappedProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "quake-unmapped";

  /// Gates the IR is restricted to, using the basis-conversion grammar. An
  /// empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Operator-norm tolerance for erasing, snapping, and synthesizing
  /// rotations. It stays text so that `CompileTarget::createFromConfig` can
  /// replace a target-argument placeholder in it, and it is empty when unset.
  std::string approximationTolerance = {};

  /// Without a cost model, the stage runs no optimization.
  std::optional<CostModel> costModel = std::nullopt;

  /// Whether reaching the basis requires Clifford+T synthesis, which is when
  /// the cost model is `t-count` and a non-empty basis cannot express an
  /// uncontrolled rotation.
  bool requiresSynthesis() const;

  bool operator==(const QuakeUnmappedProfile &other) const = default;
};

} // namespace cudaq::config

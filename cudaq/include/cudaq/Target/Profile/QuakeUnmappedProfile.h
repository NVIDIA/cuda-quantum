/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::config {

/// What the optimizer of a `quake-unmapped` stage minimizes.
enum class CostModel {
  /// Minimize T gates and non-Clifford rotations, which dominate the cost on
  /// fault-tolerant hardware.
  t_count,
  /// Minimize two-qubit gates, which dominate the error and routing cost on
  /// near-term hardware.
  two_qubit_gate_count,
};

/// IR profile for Quake before qubit mapping. Qubits come from allocations
/// rather than a device's wire set, and any two may interact. Wires and
/// references are both allowed.
///
/// Use it for a target that needs a fixed gate set but not a placement of
/// qubits on a device, such as a fault-tolerant target that counts logical
/// resources or hardware where any two qubits can interact, or before a
/// `quake-mapped` stage. With a cost model, the stage optimizes for that cost,
/// and with `t-count` and a basis that has no rotation gate it synthesizes
/// each rotation within the tolerance. Without a cost model, the stage only
/// converts to the basis. For example, a Clifford+T target lists:
///
/// \code{.yaml}
///   ir-output-profile:
///     name: quake-unmapped
///     options:
///       basis-gates: [h, s, t, x, y, z, swap, "x(1)", "y(1)", "z(1)"]
///       approximation-tolerance: 1e-10
///       cost-model: t-count
/// \endcode
struct QuakeUnmappedProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "quake-unmapped";

  /// Gates the IR is restricted to, using the basis-conversion grammar. An
  /// empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Largest operator-norm error the stage may introduce per rotation. From
  /// optimization level 1, rotations smaller than this are dropped. From level
  /// 2, rotations this close to a Clifford+T angle become Clifford+T gates.
  /// Synthesis approximates each remaining rotation within it, so a smaller
  /// tolerance costs more T gates and compile time. Zero accepts only exact
  /// rewrites, so synthesis needs a positive tolerance.
  double approximationTolerance = 0;

  /// Without a cost model, the stage runs no optimization.
  std::optional<CostModel> costModel = std::nullopt;

  bool operator==(const QuakeUnmappedProfile &other) const = default;
};

} // namespace cudaq::config

template <>
struct std::hash<cudaq::config::QuakeUnmappedProfile> {
  std::size_t
  operator()(const cudaq::config::QuakeUnmappedProfile &profile) const noexcept;
};

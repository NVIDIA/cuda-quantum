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
  /// noisy hardware.
  two_qubit_gate_count,
};

/// How the consumer of a `quake-unmapped` stage requires qubits to be
/// represented.
enum class Semantics {
  /// References or wires, as the stage chooses.
  any,
  /// Only wires.
  value,
};

/// Which loops a `quake-unmapped` stage unrolls when it converts to wires.
enum class LoopUnrolling {
  /// Every loop with a constant trip count.
  all,
  /// Only loops whose induction variable is used in the body, such as a loop
  /// that indexes a register or sets a rotation angle. A loop that repeats the
  /// same body stays rolled, so that a consumer can count it once and multiply
  /// by its trip count.
  index_use,
};

/// IR profile for Quake before qubit mapping. Qubits come from allocations
/// rather than a device's wire set, and any two may interact. Wires and
/// references are both allowed, unless `semantics` requires wires.
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

  /// With `Semantics::value`, the stage converts to wires at every level and
  /// fails to compile a kernel that keeps a reference, such as a register
  /// indexed by a loop longer than `maxUnrollIterations` or by a value known
  /// only when the kernel runs.
  Semantics semantics = Semantics::any;

  /// Which loops the stage unrolls when it converts to wires. Unless this is
  /// `all`, prep and deploy unroll no loop before the stage runs, so that the
  /// loops the stage keeps reach its output rolled.
  LoopUnrolling unrollLoops = LoopUnrolling::all;

  /// The most iterations of a loop that the stage unrolls. A selected loop
  /// with more iterations stays rolled, and the qubits it picks stay
  /// references. The default, 1024, is the limit to which prep and deploy
  /// unroll in `classical-optimization-pipeline`. A lower limit keeps prep and
  /// deploy from unrolling, as `LoopUnrolling::index_use` does.
  unsigned maxUnrollIterations = 1024;

  /// Whether prep erases `quake.apply_noise`. A consumer that has to see noise,
  /// for example to reject it, sets this to false.
  bool eraseNoise = true;

  /// Whether prep erases the QEC detector and observable operations.
  bool eraseQEC = true;

  bool operator==(const QuakeUnmappedProfile &other) const = default;
};

} // namespace cudaq::config

template <>
struct std::hash<cudaq::config::QuakeUnmappedProfile> {
  std::size_t
  operator()(const cudaq::config::QuakeUnmappedProfile &profile) const noexcept;
};

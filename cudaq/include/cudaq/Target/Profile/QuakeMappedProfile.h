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

/// IR profile for Quake after qubit mapping. Every qubit is a wire borrowed
/// from a `quake.wire_set` whose identities are device qubits, and each
/// two-qubit gate acts on a pair the wire set's coupling map connects. It
/// follows a `quake-unmapped` stage whose basis has no gate on more than two
/// qubits. For example:
///
/// \code{.yaml}
///   ir-output-profile:
///     name: quake-mapped
///     options:
///       basis-gates: [h, rx, ry, rz, x, "z(1)"]
///       device: grid(3,3)
/// \endcode
struct QuakeMappedProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "quake-mapped";

  /// Gates the IR is restricted to, using the basis-conversion grammar. An
  /// empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// The device topology in the `qubit-mapping` grammar, such as `path(N)`,
  /// `grid(w,h)`, or `file(<path>)`. Without one, nothing is routed. The
  /// target's hash covers a `file(...)` path, not the file's contents.
  std::optional<std::string> device = std::nullopt;

  bool operator==(const QuakeMappedProfile &other) const = default;
};

} // namespace cudaq::config

template <>
struct std::hash<cudaq::config::QuakeMappedProfile> {
  std::size_t
  operator()(const cudaq::config::QuakeMappedProfile &profile) const noexcept;
};

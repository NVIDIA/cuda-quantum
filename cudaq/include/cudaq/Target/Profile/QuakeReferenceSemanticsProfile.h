/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>
#include <vector>

namespace cudaq::config {

/// Constraints on measurement operations enforced by the reference-semantics
/// profile.
enum class MeasurementForm {
  /// Any measurement is allowed.
  any,
  /// Each measurement operates on exactly one `!quake.ref`.
  single_qubit,
};

/// IR profile for reference-semantics Quake IR, independent of whether the
/// represented qubits are logical, physical, or anything in between.
///
/// Value semantics Quake is currently not supported.
///
/// `quake-unmapped` supersedes this profile. This profile will be deprecated,
/// and the targets that use it moved to `quake-unmapped`, once
/// `quake-unmapped` has a cost model that counts two-qubit gates.
struct QuakeReferenceSemanticsProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "quake-reference-semantics";

  /// Gates the IR is restricted to, using the basis-conversion grammar
  /// (e.g. `rx`, `x(1)`). An empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Required measurement form.
  MeasurementForm measurementType = MeasurementForm::any;

  /// Whether a measured qubit must be reset before any subsequent
  /// non-measurement quantum use.
  bool resetAfterMeasure = false;

  bool operator==(const QuakeReferenceSemanticsProfile &other) const = default;
};

} // namespace cudaq::config

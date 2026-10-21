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

/// Constraints on measurement operations enforced by the physical IR profile.
enum class MeasurementForm {
  /// Any measurement is allowed.
  any,
  /// Each measurement operates on exactly one `!quake.ref`.
  single_qubit,
};

/// IR profile for physical, reference-semantics Quake IR.
///
/// Value semantics Quake is currently not supported.
struct PhysicalQuakeProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "physical-quake";

  /// Gates the IR is restricted to, using the basis-conversion grammar
  /// (e.g. `rx`, `x(1)`). An empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Required measurement form.
  MeasurementForm measurementType = MeasurementForm::any;

  /// Whether a measured qubit must be reset before any subsequent
  /// non-measurement quantum use.
  bool resetAfterMeasure = false;

  bool operator==(const PhysicalQuakeProfile &other) const = default;
};

} // namespace cudaq::config

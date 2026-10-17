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

/// Measurement shape required by a physical IR profile.
///
/// This constrains measurement shape only; `any` is not permission to
/// violate other profile constraints. The unimplemented `combined` form is
/// rejected when parsing a profile description.
enum class MeasurementForm {
  /// No constraint on measurement shape.
  any,
  /// Each measurement operates on exactly one `!quake.ref`.
  expanded,
};

/// IR profile for physical, reference-semantics Quake IR.
///
/// Wire/value-form Quake is not part of this profile yet.
struct PhysicalQuakeProfile {
  /// Name identifying this profile in target configurations.
  static constexpr char name[] = "physical-quake";

  /// Gates the IR is restricted to, using the basis-conversion grammar
  /// (e.g. `rx`, `x(1)`). An empty list imposes no gate-basis restriction.
  std::vector<std::string> basisGates = {};

  /// Required measurement form.
  MeasurementForm measurements = MeasurementForm::any;

  /// Whether a measured qubit must be reset before any subsequent
  /// non-measurement quantum use.
  bool resetBeforeReuse = false;

  bool operator==(const PhysicalQuakeProfile &other) const = default;
};

} // namespace cudaq::config

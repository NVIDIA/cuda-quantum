/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Yaml/v2/ProfileYaml.h"
#include "cudaq/Target/Profile/QuakeReferenceSemanticsProfile.h"
#include <string>
#include <vector>

template <>
struct rfl::Reflector<cudaq::config::MeasurementForm> {
  using ReflType = std::string;
  static cudaq::config::MeasurementForm to(const std::string &name) {
    return cudaq::config::detail::enumFromYamlName<
        cudaq::config::MeasurementForm>(name);
  }
  static std::string from(const cudaq::config::MeasurementForm &value) {
    switch (value) {
    case cudaq::config::MeasurementForm::any:
      return "any";
    case cudaq::config::MeasurementForm::single_qubit:
      return "single-qubit";
    }
    cudaq::config::detail::unknownEnumerator();
  }
};

/// YAML form of `QuakeReferenceSemanticsProfile`. Every option may be omitted
/// and then takes the profile's default.
template <>
struct cudaq::config::v2::ProfileYaml<
    cudaq::config::QuakeReferenceSemanticsProfile> {
  struct Yaml {
    using Tag = rfl::Literal<QuakeReferenceSemanticsProfile::name>;
    struct Options {
      rfl::DefaultVal<std::vector<std::string>> basisGates =
          QuakeReferenceSemanticsProfile{}.basisGates;
      rfl::DefaultVal<MeasurementForm> measurementType =
          QuakeReferenceSemanticsProfile{}.measurementType;
      rfl::DefaultVal<bool> resetAfterMeasure =
          QuakeReferenceSemanticsProfile{}.resetAfterMeasure;
    };
    rfl::DefaultVal<Options> options;
  };
  // Checks that the options and the profile have the same number of fields,
  // so that a profile field without a YAML option fails the build.
  static_assert(rfl::named_tuple_t<Yaml::Options>::size() ==
                rfl::named_tuple_t<QuakeReferenceSemanticsProfile>::size());

  static QuakeReferenceSemanticsProfile to(const Yaml &yaml);
  static Yaml from(const QuakeReferenceSemanticsProfile &profile);
};

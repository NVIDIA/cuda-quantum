/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Yaml/v2/ProfileYaml.h"
#include "cudaq/Target/Profile/QuakeUnmappedProfile.h"
#include <optional>
#include <string>
#include <vector>

template <>
struct rfl::Reflector<cudaq::config::CostModel> {
  using ReflType = std::string;
  static cudaq::config::CostModel to(const std::string &name) {
    return cudaq::config::detail::enumFromYamlName<cudaq::config::CostModel>(
        name);
  }
  static std::string from(const cudaq::config::CostModel &value) {
    switch (value) {
    case cudaq::config::CostModel::t_count:
      return "t-count";
    }
    cudaq::config::detail::unknownEnumerator();
  }
};

/// YAML form of `QuakeUnmappedProfile`. Every option may be omitted and then
/// takes the profile's default.
template <>
struct cudaq::config::v2::ProfileYaml<cudaq::config::QuakeUnmappedProfile> {
  struct Yaml {
    using Tag = rfl::Literal<QuakeUnmappedProfile::name>;
    struct Options {
      rfl::DefaultVal<std::vector<std::string>> basisGates =
          QuakeUnmappedProfile{}.basisGates;
      rfl::DefaultVal<double> approximationTolerance =
          QuakeUnmappedProfile{}.approximationTolerance;
      std::optional<CostModel> costModel;
    };
    rfl::DefaultVal<Options> options;
  };
  // Checks that the options and the profile have the same number of fields,
  // so that a profile field without a YAML option fails the build.
  static_assert(rfl::named_tuple_t<Yaml::Options>::size() ==
                rfl::named_tuple_t<QuakeUnmappedProfile>::size());

  static QuakeUnmappedProfile to(const Yaml &yaml);
  static Yaml from(const QuakeUnmappedProfile &profile);
};

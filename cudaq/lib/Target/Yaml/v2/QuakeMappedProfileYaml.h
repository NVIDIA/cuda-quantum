/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Yaml/v2/ProfileYaml.h"
#include "cudaq/Target/Profile/QuakeMappedProfile.h"
#include <optional>
#include <string>
#include <vector>

/// YAML form of `QuakeMappedProfile`. Every option may be omitted and then
/// takes the profile's default.
template <>
struct cudaq::config::v2::ProfileYaml<cudaq::config::QuakeMappedProfile> {
  struct Yaml {
    using Tag = rfl::Literal<QuakeMappedProfile::name>;
    struct Options {
      rfl::DefaultVal<std::vector<std::string>> basisGates =
          QuakeMappedProfile{}.basisGates;
      std::optional<std::string> device;
    };
    rfl::DefaultVal<Options> options;
  };
  // Checks that the options and the profile have the same number of fields,
  // so that a profile field without a YAML option fails the build.
  static_assert(rfl::named_tuple_t<Yaml::Options>::size() ==
                rfl::named_tuple_t<QuakeMappedProfile>::size());

  static QuakeMappedProfile to(const Yaml &yaml);
  static Yaml from(const QuakeMappedProfile &profile);
};

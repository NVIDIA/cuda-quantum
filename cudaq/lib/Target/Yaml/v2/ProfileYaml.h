/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Yaml/TargetConfigYamlAdapters.h"
#include <variant>

namespace cudaq::config::v2 {

/// The YAML form of IR profile `P`. Each profile specializes it in its own
/// file with a `Yaml` type, whose `Tag` is the profile's name, and conversions
/// `to` the profile and `from` it. The specializations are a project trait
/// rather than `rfl::Reflector<P>`, so the public profile types gain no
/// reflect-cpp serialization.
template <typename P>
struct ProfileYaml;

/// Union of the YAML forms of the profiles in variant `V`, selected by `name`.
template <typename V>
struct TaggedProfiles;
template <typename... P>
struct TaggedProfiles<std::variant<P...>> {
  using type = rfl::TaggedUnion<"name", typename ProfileYaml<P>::Yaml...>;

  /// Converts the parsed alternative with its profile's `ProfileYaml`. The
  /// YAML forms are distinct types, so overload resolution selects the
  /// profile.
  static std::variant<P...> toProfile(const type &yaml) {
    struct FromYaml : ProfileYaml<P>... {
      using ProfileYaml<P>::to...;
    };
    return yaml.visit([](const auto &alternative) -> std::variant<P...> {
      return FromYaml::to(alternative);
    });
  }
};

} // namespace cudaq::config::v2

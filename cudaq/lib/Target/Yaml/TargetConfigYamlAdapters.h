/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include <algorithm>
#include <rfl.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cudaq::config {

/// Maps camelCase C++ field names onto the kebab-case YAML vocabulary and
/// rejects unknown fields.
using YamlProcessors =
    rfl::Processors<rfl::CamelCaseToSnakeCase, rfl::SnakeCaseToKebabCase,
                    rfl::NoExtraFields>;

namespace detail {

/// Matches a YAML enum name against the reflected enumerator names with
/// underscores replaced by dashes (e.g. `dep_analysis` -> `dep-analysis`).
/// Unmatched (including numeric) spellings are rejected.
template <typename E>
  requires std::is_enum_v<E>
E enumFromYamlName(const std::string &name) {
  std::string allowed;
  for (const auto &[enumName, value] : rfl::get_enumerator_array<E>()) {
    std::string dashed{enumName};
    std::replace(dashed.begin(), dashed.end(), '_', '-');
    if (dashed == name)
      return value;
    if (!allowed.empty())
      allowed += ", ";
    allowed += dashed;
  }
  throw std::runtime_error("unknown value '" + name + "' (allowed: " + allowed +
                           ")");
}

template <typename E>
  requires std::is_enum_v<E>
std::string enumToYamlName(E value) {
  for (const auto &[enumName, enumerator] : rfl::get_enumerator_array<E>())
    if (enumerator == value) {
      std::string dashed{enumName};
      std::replace(dashed.begin(), dashed.end(), '_', '-');
      return dashed;
    }
  throw std::runtime_error("cannot serialize unknown enumerator value");
}

} // namespace detail

} // namespace cudaq::config

namespace rfl {

template <>
struct Reflector<cudaq::config::TargetFeatureFlag> {
  using ReflType = std::string;
  static cudaq::config::TargetFeatureFlag to(const std::string &name) {
    return cudaq::config::detail::enumFromYamlName<
        cudaq::config::TargetFeatureFlag>(name);
  }
  static std::string from(const cudaq::config::TargetFeatureFlag &value) {
    return cudaq::config::detail::enumToYamlName(value);
  }
};

template <>
struct Reflector<cudaq::config::ArgumentType> {
  using ReflType = std::string;
  static cudaq::config::ArgumentType to(const std::string &name) {
    return cudaq::config::detail::enumFromYamlName<cudaq::config::ArgumentType>(
        name);
  }
  static std::string from(const cudaq::config::ArgumentType &value) {
    return cudaq::config::detail::enumToYamlName(value);
  }
};

} // namespace rfl

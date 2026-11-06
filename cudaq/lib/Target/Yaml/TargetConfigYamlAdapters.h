/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
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

/// Matches a YAML enum name against the names that `rfl::Reflector<E>::from`
/// gives each enumerator. Unmatched (including numeric) spellings are
/// rejected.
template <typename E>
  requires std::is_enum_v<E>
E enumFromYamlName(const std::string &name) {
  std::string allowed;
  for (const auto &enumerator : rfl::get_enumerator_array<E>()) {
    const std::string yamlName = rfl::Reflector<E>::from(enumerator.second);
    if (yamlName == name)
      return enumerator.second;
    if (!allowed.empty())
      allowed += ", ";
    allowed += yamlName;
  }
  throw std::runtime_error("unknown value '" + name + "' (allowed: " + allowed +
                           ")");
}

[[noreturn]] inline void unknownEnumerator() {
  throw std::runtime_error("cannot serialize unknown enumerator value");
}

} // namespace detail

} // namespace cudaq::config

// Each `from` spells the YAML names, so that renaming an enumerator does not
// change the YAML. A missing case is a `-Wswitch` warning.
namespace rfl {

template <>
struct Reflector<cudaq::config::TargetFeatureFlag> {
  using ReflType = std::string;
  static cudaq::config::TargetFeatureFlag to(const std::string &name) {
    return cudaq::config::detail::enumFromYamlName<
        cudaq::config::TargetFeatureFlag>(name);
  }
  static std::string from(const cudaq::config::TargetFeatureFlag &value) {
    using Flag = cudaq::config::TargetFeatureFlag;
    switch (value) {
    case Flag::fp32:
      return "fp32";
    case Flag::fp64:
      return "fp64";
    case Flag::mgpu:
      return "mgpu";
    case Flag::mqpu:
      return "mqpu";
    case Flag::dep_analysis:
      return "dep-analysis";
    case Flag::qpp:
      return "qpp";
    }
    cudaq::config::detail::unknownEnumerator();
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
    using Type = cudaq::config::ArgumentType;
    switch (value) {
    case Type::string:
      return "string";
    case Type::integer:
      return "integer";
    case Type::uuid:
      return "uuid";
    case Type::option_flags:
      return "option-flags";
    case Type::machine_config:
      return "machine-config";
    }
    cudaq::config::detail::unknownEnumerator();
  }
};

} // namespace rfl

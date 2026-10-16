/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/CanonicalConversion.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <type_traits>

namespace cudaq::config::detail {

std::string trimScalar(const std::string &segment) {
  const auto isSpace = [](unsigned char c) { return std::isspace(c); };
  const auto begin = std::find_if_not(segment.begin(), segment.end(), isSpace);
  const auto end =
      std::find_if_not(segment.rbegin(), segment.rend(), isSpace).base();
  if (begin >= end)
    return {};
  return std::string(begin, end);
}

std::vector<std::string> normalizeSimulationBackends(
    const rfl::Variant<std::string, std::vector<std::string>> &input) {
  return input.visit([](const auto &value) -> std::vector<std::string> {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      return value;
    } else {
      std::vector<std::string> result;
      std::size_t begin = 0;
      while (begin <= value.size()) {
        const auto end = value.find(',', begin);
        auto segment = trimScalar(value.substr(begin, end - begin));
        if (!segment.empty())
          result.push_back(std::move(segment));
        if (end == std::string::npos)
          break;
        begin = end + 1;
      }
      return result;
    }
  });
}

void validatePattern(const std::string &pattern) {
  try {
    std::regex re(pattern);
  } catch (const std::regex_error &e) {
    throw std::runtime_error("'" + pattern +
                             "' is not a valid regex: " + e.what());
  }
}

} // namespace cudaq::config::detail

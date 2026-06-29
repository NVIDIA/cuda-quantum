/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::utils::anyon {

/// @brief Combine the per-register shot results from an Anyon QPU response into
/// a single vector of concatenated bitstrings (one entry per shot).
inline std::vector<std::string>
combineRegisterResults(const nlohmann::json &results,
                       const std::vector<std::string> &orderedRegisterNames) {
  if (orderedRegisterNames.empty())
    return {};

  auto nShots = results.begin().value().get<std::vector<std::string>>().size();
  std::vector<std::string> bitstrings(nShots);
  for (const auto &registerName : orderedRegisterNames) {
    auto bitResults = results.at(registerName).get<std::vector<std::string>>();
    if (bitResults.size() != nShots)
      throw std::runtime_error("Inconsistent shot count in results: expected " +
                               std::to_string(nShots) + " but register '" +
                               registerName + "' has " +
                               std::to_string(bitResults.size()) + ".");
    for (std::size_t i = 0; auto &bit : bitResults)
      bitstrings[i++] += bit;
  }
  return bitstrings;
}

} // namespace cudaq::utils::anyon

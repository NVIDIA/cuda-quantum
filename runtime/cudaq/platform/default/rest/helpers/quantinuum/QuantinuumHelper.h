/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/SampleResult.h"
#include "nlohmann/json.hpp"
#include <bitset>

namespace cudaq::utils::quantinuum {

// Helper to process Quantinuum Nexus result
// Note: we split this out in a header file for testability purposes.
inline cudaq::sample_result
processResults(const nlohmann::json &shotResultJson,
               const std::vector<std::string> &registerNames) {
  // Helper to process Nexus outcome arrays (arrays of bytes) into bitstrings
  const auto outComeArrayToBitString =
      [](const std::vector<int8_t> &outcomeArray,
         std::size_t numBits) -> std::string {
    const std::size_t expectedArrayLength = (numBits + 7) / 8;
    if (outcomeArray.size() != expectedArrayLength) {
      throw std::runtime_error("Outcome array size does not match expected "
                               "length based on number of bits: " +
                               std::to_string(numBits) + " vs " +
                               std::to_string(outcomeArray.size() * 8));
    }

    std::string bitString(numBits, '0');
    int bitIndex = numBits - 1; // high to low indexing
    for (auto byte : outcomeArray) {
      std::bitset<8> bits(byte);
      for (int i = 7; i >= 0; --i) {
        const auto bit = bits[i];
        bitString[bitIndex--] = '0' + bit;
        if (bitIndex < 0) {
          break; // We have filled the bit string
        }
      }
    }

    return bitString;
  };

  if (!shotResultJson.contains("width") || !shotResultJson.contains("array")) {
    throw std::runtime_error(
        "Invalid shot result JSON: missing 'width' or 'array' keys.");
  }

  const std::size_t numBits = shotResultJson["width"].get<std::size_t>();
  const std::vector<std::vector<int8_t>> shotsOutcomeArray =
      shotResultJson["array"].get<std::vector<std::vector<int8_t>>>();
  const auto numShots = shotsOutcomeArray.size();

  // The names are listed in the reverse order (w.r.t. CUDA-Q bit indexing
  // convention)
  std::vector<CountsDictionary> registerResults(registerNames.size());
  std::vector<std::vector<std::string>> registerSequentialData(
      registerNames.size());
  for (auto &data : registerSequentialData) {
    data.reserve(numShots);
  }
  cudaq::CountsDictionary globalCounts;
  std::vector<std::string> globalSequentialData;
  globalSequentialData.reserve(numShots);
  for (const auto &outcomeArray : shotsOutcomeArray) {
    // Convert the outcome array to a bit string
    const auto bitString = outComeArrayToBitString(outcomeArray, numBits);
    assert(bitString.length() == registerNames.size());
    // Populate the register results
    for (std::size_t i = 0; i < registerNames.size(); ++i) {
      const auto bit = bitString[i];
      registerResults[i][std::string{bit}]++;
      registerSequentialData[i].push_back(std::string{bit});
    }
    // Global register results
    globalCounts[bitString]++;
    globalSequentialData.push_back(bitString);
  }

  std::vector<cudaq::ExecutionResult> allResults;
  allResults.reserve(registerNames.size() + 1);
  for (std::size_t i = 0; i < registerNames.size(); ++i) {
    allResults.push_back({registerResults[i], registerNames[i]});
    allResults.back().sequentialData = registerSequentialData[i];
  }

  // Add the global register results
  cudaq::ExecutionResult result{globalCounts, GlobalRegisterName};
  result.sequentialData = globalSequentialData;
  allResults.push_back(result);
  return cudaq::sample_result{allResults};
}

} // namespace cudaq::utils::quantinuum

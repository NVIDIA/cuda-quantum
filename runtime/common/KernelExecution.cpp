/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "KernelExecution.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace cudaq {

void validateExecutionMetadata(const ActiveDeviceQubits &activeDeviceQubits,
                               const nlohmann::json &outputNames) {
  if (!std::is_sorted(activeDeviceQubits.begin(), activeDeviceQubits.end()))
    throw std::invalid_argument("active device qubits must be sorted");
  if (std::adjacent_find(activeDeviceQubits.begin(),
                         activeDeviceQubits.end()) != activeDeviceQubits.end())
    throw std::invalid_argument("active device qubits must be unique");

  if (outputNames.is_null() || outputNames.empty())
    return;

  std::unordered_set<std::size_t> outputPositions;
  std::size_t resultIndex = 0;
  for (const auto &entry : outputNames.at(0)) {
    const auto &outputLocation = entry.at(1);
    // The third tuple element is the user-visible output position. An old
    // compiler omits it, in which case fall back to the result index.
    std::size_t outputPosition = resultIndex;
    if (outputLocation.size() > 2)
      outputPosition = outputLocation.at(2).get<std::size_t>();
    if (!outputPositions.insert(outputPosition).second)
      throw std::invalid_argument(
          "output_names contains duplicate output positions");
    ++resultIndex;
  }
  if (!outputPositions.empty() &&
      *std::max_element(outputPositions.begin(), outputPositions.end()) !=
          outputPositions.size() - 1)
    throw std::invalid_argument("output_names output positions must be dense");
}

KernelExecution::KernelExecution(const std::string &n, const std::string &c,
                                 std::optional<cudaq::JitEngine> jit,
                                 std::optional<Resources> rc)
    : name(n), code(c), jit(jit), resourceCounts(rc),
      output_names(nlohmann::json{}), user_data(nlohmann::json{}) {}
KernelExecution::KernelExecution(const std::string &n, const std::string &c,
                                 std::optional<cudaq::JitEngine> jit,
                                 std::optional<Resources> rc, nlohmann::json &o)
    : name(n), code(c), jit(jit), resourceCounts(rc), output_names(o),
      user_data(nlohmann::json{}) {}

KernelExecution::~KernelExecution() = default;

} // namespace cudaq

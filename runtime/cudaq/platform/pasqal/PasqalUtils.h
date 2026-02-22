/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudaq::pasqal {

inline ExecutionResult parseExecutionResult(const nlohmann::json &payload) {
  if (!payload.is_object())
    throw std::runtime_error(
        std::string("Invalid JSON object received as job result."));

  CountsDictionary counts;
  for (auto &[bitstring, count] : payload.items()) {
    auto littleEndianBitstring = bitstring;
    std::reverse(littleEndianBitstring.begin(), littleEndianBitstring.end());
    counts[littleEndianBitstring] = count.get<std::size_t>();
  }

  return ExecutionResult(counts);
}

inline ExecutionResult
parseExecutionResultFromTaskResult(const std::string &taskResultJson) {
  auto payload = nlohmann::json::parse(taskResultJson);

  // Fix this outside CUDA-Q. Why are these payloads different?
  if (payload.contains("counter") && payload["counter"].is_object()) {
    payload = payload["counter"];
  }
  return parseExecutionResult(payload);
}

} // namespace cudaq::pasqal

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace cudaq::qio {
    static
    QuantumProgramResult
    QuantumProgramResult::fromJson(nlohmann::json json) const {

    }

    std::vector<cudaq::ExecutionResult>
    QuantumProgramResult::toExecutionResults() const {
      cudaq::CountsDictionary counts;

      for (const auto& sample : result.getSamples()) {
          std::string bitString;
          for (auto bit : sample.bits) {
              bitString += std::to_string(bit);
          }
          counts[bitString] += 1;
      }

      std::vector<ExecutionResult> execResults;
      execResults.emplace_back(ExecutionResult{counts});
    }
}

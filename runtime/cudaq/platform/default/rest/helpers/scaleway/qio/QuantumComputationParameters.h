/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace cudaq::qio {
class QuantumComputationParameters {
public:
  explicit QuantumComputationParameters(std::size_t shots,
                                        nlohmann::json options);

  nlohmann::json toJson() const;

  static QuantumComputationParameters fromJson(nlohmann::json json);

  nlohmann::json options();

private:
  std::size_t m_shots;
  nlohmann::json m_options;
};
} // namespace cudaq::qio

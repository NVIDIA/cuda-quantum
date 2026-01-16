/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <string>

namespace qio {

class QuantumComputationParameters {
public:
  explicit QuantumComputationParameters(std::size_t shots);

  nlohmann::json toJson() const;

private:
  std::size_t m_shots;
};

}

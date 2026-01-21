/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumComputationParameters.h"

using json = nlohmann::json;

namespace cudaq::qio {
  QuantumComputationParameters::QuantumComputationParameters(
    std::size_t shots,
    std::unordered_map<std::string, std::string> options)
      : m_shots(shots),
        m_options(options) {}

  json
  QuantumComputationParameters::toJson() const {
    return {
        {"shots", m_shots},
        {"options", m_options}
    };
  }
}

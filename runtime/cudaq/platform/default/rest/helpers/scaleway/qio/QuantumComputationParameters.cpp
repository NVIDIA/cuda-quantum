/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumComputationParameters.h"

using json = nlohmann::json;

namespace qio {

QuantumComputationParameters::QuantumComputationParameters(std::size_t shots)
    : m_shots(shots) {}

json QuantumComputationParameters::toJson() const {
  return {
      {"shots", m_shots}
  };
}

}

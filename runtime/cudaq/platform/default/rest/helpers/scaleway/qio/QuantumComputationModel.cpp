/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumComputationModel.h"

using json = nlohmann::json;

namespace cudaq::qio {

QuantumComputationModel::QuantumComputationModel(
    std::vector<QuantumProgram> programs)
    : m_programs(std::move(programs)) {}

json
QuantumComputationModel::toJson() const {
  json programsJson = json::array();
  for (const auto &p : m_programs) {
    programsJson.push_back(p.toJson());
  }

  return {
      {"programs", programsJson},
  };
}
}

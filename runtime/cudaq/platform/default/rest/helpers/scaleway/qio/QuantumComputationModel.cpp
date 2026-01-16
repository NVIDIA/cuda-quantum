/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "quantum_computation_model.h"

using json = nlohmann::json;

namespace qio {

QuantumComputationModel::QuantumComputationModel(
    QioQuantumProgram program,
    QuantumComputationParameters parameters)
    : m_program(std::move(program)),
      m_parameters(std::move(parameters)) {}

json QuantumComputationModel::toJson() const {
  return {
      {"program", m_program.toJson()},
      {"parameters", m_parameters.toJson()}
  };
}

}

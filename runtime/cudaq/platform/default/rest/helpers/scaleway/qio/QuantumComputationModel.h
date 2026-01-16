/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <nlohmann/json.hpp>

#include "QuantumProgram.h"
#include "QuantumComputationParameters.h"

namespace qio {

class QioQuantumComputationModel {
public:
  QioQuantumComputationModel(std::vector<QioQuantumProgram> programs,
                             QuantumComputationParameters parameters);

  nlohmann::json toJson() const;

private:
  std::vector<QioQuantumProgram> m_programs;
  QuantumComputationParameters m_parameters;
};

}

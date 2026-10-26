/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "Compression.h"
#include "Format.h"
#include <nlohmann/json.hpp>
#include <string>

namespace cudaq::qio {
class QuantumProgram {
public:
  QuantumProgram(const std::string &serialization,
                 QuantumProgramSerializationFormat serializationFormat,
                 CompressionFormat compressionFormat);

  nlohmann::json toJson() const;

private:
  std::string m_serialization;
  QuantumProgramSerializationFormat m_serializationFormat;
  CompressionFormat m_compressionFormat;
};
} // namespace cudaq::qio

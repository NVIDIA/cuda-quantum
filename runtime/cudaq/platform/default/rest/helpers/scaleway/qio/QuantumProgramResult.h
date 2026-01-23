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
#include "common/SampleResult.h"
#include <nlohmann/json.hpp>
#include <string>

namespace cudaq::qio {
class QuantumProgramResult {
public:
  QuantumProgramResult(std::string serialization,
                      QuantumProgramResultSerializationFormat serializationFormat,
                      CompressionFormat compressionFormat);

  static QuantumProgramResult fromJson(nlohmann::json json);

  cudaq::sample_result toCudaqSampleResult();

private:
  std::string m_serialization;
  QuantumProgramResultSerializationFormat m_serializationFormat;
  CompressionFormat m_compressionFormat;
};
} // namespace cudaq::qio

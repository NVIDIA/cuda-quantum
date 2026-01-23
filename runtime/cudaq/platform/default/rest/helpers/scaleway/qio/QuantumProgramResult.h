/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "Compression.h"
#include <nlohmann/json.hpp>
#include <string>

namespace cudaq::qio {
class QuantumProgramResult {
public:
  enum class SerializationFormat {
    UNKOWN_SERIALIZATION_FORMAT = 0,
    CIRQ_RESULT_JSON_V1 = 1,
    QISKIT_RESULT_JSON_V1 = 2,
    CUDAQ_SAMPLE_RESULT_JSON_V1 = 3
  };

  enum class CompressionFormat {
    UNKNOWN_COMPRESSION_FORMAT = 0,
    NONE = 1,
    ZLIB_BASE64_V1 = 2
  };

  static QuantumProgramResult fromJson(nlohmann::json json);

  cudaq::sample_result toCudaqSampleResult();

private:
  std::string m_serialization;
  SerializationFormat m_serializationFormat;
  CompressionFormat m_compressionFormat;
};
} // namespace cudaq::qio

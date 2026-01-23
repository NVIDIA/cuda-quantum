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
class QuantumProgram {
public:
  enum SerializationFormat {
    UNKOWN_SERIALIZATION_FORMAT = 0,
    QASM_V1 = 1,
    QASM_V2 = 2,
    QASM_V3 = 3,
    QIR_V1 = 4
  };

  enum CompressionFormat {
    UNKNOWN_COMPRESSION_FORMAT = 0,
    NONE = 1,
    ZLIB_BASE64_V1 = 2
  };

  QuantumProgram(const cudaq::Kernel &kernel,
                 SerializationFormat serializationFormat,
                 CompressionFormat compressionFormat);

  QuantumProgram(const std::string &serialization,
                 SerializationFormat serializationFormat,
                 CompressionFormat compressionFormat);

  nlohmann::json toJson() const;

private:
  std::string m_serialization;
  SerializationFormat m_serializationFormat;
  CompressionFormat m_compressionFormat;
};
} // namespace cudaq::qio

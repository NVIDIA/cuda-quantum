/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumProgram.h"
#include "Base64.h"
#include "Compression.h"

using json = nlohmann::json;
using namespace cudaq::qio;

QuantumProgram::QuantumProgram(const std::string &serialization,
                               QuantumProgramSerializationFormat serializationFormat,
                               CompressionFormat compressionFormat)
    : m_serializationFormat(serializationFormat),
      m_compressionFormat(compressionFormat) {
        if (m_compressionFormat == CompressionFormat::ZLIB_BASE64_V1) {
          std::string compressedSerialization =
              gzipCompress(serialization);
          m_serialization = encodeBase64(compressedSerialization);
        } else if (m_compressionFormat == CompressionFormat::NONE) {
          m_serialization = serialization;
        }
    }

json QuantumProgram::toJson() const {
  return {{"serialization", m_serialization},
          {"serialization_format", m_serializationFormat},
          {"compression_format", m_compressionFormat}};
}

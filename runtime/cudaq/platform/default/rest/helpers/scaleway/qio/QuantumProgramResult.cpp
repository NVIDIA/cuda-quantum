/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "QuantumProgramResult.h"

namespace cudaq::qio {
static QuantumProgramResult
QuantumProgramResult::fromJson(nlohmann::json j) {
  return QuantumProgramResult(
      j.value("serialization", ""),
      j.value("serialization_format",
              SerializationFormat::UNKOWN_SERIALIZATION_FORMAT),
      j.value("compression_format", CompressionFormat::NONE));
}

cudaq::sample_result QuantumProgramResult::toCudaqSampleResult() const {
  if (m_serializationFormat !=
      SerializationFormat::CUDAQ_SAMPLE_RESULT_JSON_V1) {
    throw std::runtime_error("QuantumProgramResult: Unsupported serialization "
                             "format for conversion to cudaq::sample_result");
  }

  std::string uncompressedSerialization = m_serialization;

  if (m_compressionFormat
    == CompressionFormat::ZLIB_BASE64_V1) {
      std::string decodedSerialization =
          compression::base64Decode(m_serialization);
      uncompressedSerialization =
          compression::gzipDecompress(decodedSerialization);
    }
  else if (m_compressionFormat != CompressionFormat::NONE) {
    throw std::runtime_error("QuantumProgramResult: Unsupported compression "
                             "format for conversion to cudaq::sample_result");
  }

  auto resultJson = nlohmann::json::parse(uncompressedSerialization);

  cudaq::sample_result sample_result;

  sample_result.deserialize(resultJson);

  return sample_result;
}
} // namespace cudaq::qio

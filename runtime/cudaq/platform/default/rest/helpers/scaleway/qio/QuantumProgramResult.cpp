/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumProgramResult.h"
#include "Compression.h"
#include "Base64.h"
#include "common/Logger.h"

using json = nlohmann::json;
using namespace cudaq::qio;

void appendStringSerialized(const std::string &s,
                            std::vector<std::size_t> &out) {
  out.push_back(s.size());
  for (char c : s) {
    out.push_back(static_cast<std::size_t>(c));
  }
}

std::vector<std::size_t>
qiskitResultToCudaqSampleResult(
    const std::vector<
        std::pair<std::string,
                   std::unordered_map<std::string, std::size_t>>> &qiskitResult) {
  std::vector<std::size_t> serialized;

  for (const auto &exp : qiskitResult) {
    const std::string &regName = exp.first;
    const auto &counts = exp.second;

    // 1) Serialize register name
    appendStringSerialized(regName, serialized);

    // 2) Serialize counts: for each bitstring -> count
    for (const auto &kv : counts) {
      const std::string &bitstring = kv.first;
      std::size_t count = kv.second;

      appendStringSerialized(bitstring, serialized);
      serialized.push_back(count);
    }
  }

  return serialized;
}

QuantumProgramResult::QuantumProgramResult(std::string serialization,
                      QuantumProgramResultSerializationFormat serializationFormat,
                      CompressionFormat compressionFormat) :
    m_serialization(serialization),
    m_serializationFormat(serializationFormat),
    m_compressionFormat(compressionFormat) {}

QuantumProgramResult
QuantumProgramResult::fromJson(json j) {
  return QuantumProgramResult(
      j.value("serialization", ""),
      j.value("serialization_format",
              QuantumProgramResultSerializationFormat::UNKOWN_RESULT_SERIALIZATION_FORMAT),
      j.value("compression_format", CompressionFormat::NONE));
}

cudaq::sample_result QuantumProgramResult::toCudaqSampleResult() {
  std::string uncompressedSerialization = m_serialization;

  if (m_compressionFormat
    == CompressionFormat::ZLIB_BASE64_V1) {
      std::string decodedSerialization =
          decodeBase64(m_serialization);
      uncompressedSerialization =
          gzipDecompress(decodedSerialization);
    }
  else if (m_compressionFormat != CompressionFormat::NONE) {
    throw std::runtime_error("QuantumProgramResult: Unsupported compression "
                             "format for conversion to cudaq::sample_result");
  }

  cudaq::sample_result sampleResult;

  if (m_serializationFormat ==
    QuantumProgramResultSerializationFormat::CUDAQ_SAMPLE_RESULT_JSON_V1) {
      auto resultJson = json::parse(uncompressedSerialization);
    
      auto serialization = resultJson.get<std::vector<std::size_t>>();
    
      sampleResult.deserialize(serialization);
  } else if (m_serializationFormat ==
    QuantumProgramResultSerializationFormat::QISKIT_RESULT_JSON_V1) {
      auto resultJson = json::parse(uncompressedSerialization);

      CUDAQ_INFO("Get qiskit result:", resultJson);

      auto qiskitResult = resultJson.get<std::vector<std::pair<std::string,
        std::unordered_map<std::string, std::size_t>>>>();

      auto serialization = qiskitResultToCudaqSampleResult(qiskitResult);

      sampleResult.deserialize(serialization);
  } else {
    throw std::runtime_error("QuantumProgramResult: Unsupported serialization "
        "format for conversion to cudaq::sample_result");
  }
  
  return sampleResult;
}
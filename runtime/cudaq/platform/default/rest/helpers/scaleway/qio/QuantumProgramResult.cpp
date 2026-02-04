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
#include <bitset>

using json = nlohmann::json;
using namespace cudaq::qio;

struct QiskitExperimentResultData {
  // ex: {"0x3" -> 6170, "0x0" -> 6175}
  std::unordered_map<std::string, std::size_t> counts;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QiskitExperimentResultData, counts)

struct QiskitExperimentResultHeader {
  std::string name;
  int n_qubits;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QiskitExperimentResultHeader, name, n_qubits)

struct QiskitExperimentResult {
  QiskitExperimentResultData data;
  QiskitExperimentResultHeader header;
  bool success = false;
  int shots = 0;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(QiskitExperimentResult, success, data, shots)

std::string hexToBitstring(const std::string &hex, int n_qubits) {
    std::string clean = hex.substr(2);

    std::size_t value;
    std::stringstream ss;
    ss << std::hex << clean;
    ss >> value;

    std::string bits = std::bitset<64>(value).to_string();

    CUDAQ_INFO("bits : {}", bits);

    return bits.substr(64 - n_qubits, n_qubits);
}

void appendStringSerialized(const std::string &s,
                            std::vector<std::size_t> &out) {
  out.push_back(s.size());
  for (char c : s) {
    out.push_back(static_cast<std::size_t>(c));
  }
}

std::vector<std::size_t>
qiskitResultToCudaqSampleResult(QiskitExperimentResult qiskitResult) {
  std::vector<std::size_t> serialized;

  CUDAQ_INFO("qiskit result : {}, {}", qiskitResult.success, qiskitResult.shots);
  CUDAQ_INFO("qiskit result name: {}", qiskitResult.header.name);
  CUDAQ_INFO("qiskit result count: {}", qiskitResult.header.n_qubits);

  appendStringSerialized(qiskitResult.header.name, serialized);

  for (const auto &kv : qiskitResult.data.counts) {
      const std::string &hexKey = kv.first;
      std::size_t count = kv.second;
      std::string bitstring = hexToBitstring(hexKey, qiskitResult.header.n_qubits);

      CUDAQ_INFO("bitstring: {}, {}", kv.first, bitstring);

      appendStringSerialized(bitstring, serialized);
      serialized.push_back(count);
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
      CUDAQ_INFO("Get qiskit result: {}", uncompressedSerialization);
      auto qiskitResults = resultJson["results"].get<std::vector<QiskitExperimentResult>>();

      if (qiskitResults.size() == 0) {
          throw std::runtime_error("QuantumProgramResult: empty ExperimentResult");
      }

      auto serialization = qiskitResultToCudaqSampleResult(qiskitResults[0]);

      sampleResult.deserialize(serialization);
  } else {
    throw std::runtime_error("QuantumProgramResult: Unsupported serialization "
        "format for conversion to cudaq::sample_result");
  }
  
  return sampleResult;
}
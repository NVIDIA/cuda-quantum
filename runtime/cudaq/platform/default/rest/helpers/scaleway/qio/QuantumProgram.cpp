/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "quantum_program.h"
#include "compression.h"

using json = nlohmann::json;

namespace qio {

QuantumProgram::QuantumProgram(
    const cudaq::Kernel &kernel,
    SerializationFormat serializationFormat,
    CompressionFormat compressionFormat)
    : m_serializationFormat(serializationFormat),
      m_compressionFormat(compressionFormat) {

  if (m_serializationFormat != SerializationFormat::QIR) {
    throw std::runtime_error(
        "Only QIR serialization is implemented in qio");
  }

  std::string serialization = cudaq::to_qir(kernel);

  if (m_compressionFormat == CompressionFormat::ZLIB_BASE64_V1) {
    auto gz = compression::gzipCompress(serialization);
    serialization = compression::base64Encode(gz);
  } else {
    serialization = compression::base64Encode(serialization);
  }
}

json QuantumProgram::toJson() const {
  return {
      {"serialization", m_serialization},
      {"serialization_format", m_serializationFormat},
      {"compression_format", m_compressionFormat}
  };
}

}

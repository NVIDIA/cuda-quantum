/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumProgram.h"

using json = nlohmann::json;

namespace cudaq::qio {
QuantumProgram::QuantumProgram(const std::string &serialization,
                               QuantumProgramSerializationFormat serializationFormat,
                               CompressionFormat compressionFormat)
    : m_serialization(serialization),
      m_serializationFormat(serializationFormat),
      m_compressionFormat(compressionFormat) {}

json QuantumProgram::toJson() const {
  return {{"serialization", m_serialization},
          {"serialization_format", m_serializationFormat},
          {"compression_format", m_compressionFormat}};
}
} // namespace cudaq::qio

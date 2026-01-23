/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

namespace cudaq::qio {
  enum QuantumProgramSerializationFormat {
    UNKOWN_SERIALIZATION_FORMAT = 0,
    QASM_V1 = 1,
    QASM_V2 = 2,
    QASM_V3 = 3,
    QIR_V1 = 4
  };

  enum QuantumProgramResultSerializationFormat {
    UNKOWN_SERIALIZATION_FORMAT = 0,
    CIRQ_RESULT_JSON_V1 = 1,
    QISKIT_RESULT_JSON_V1 = 2,
    CUDAQ_SAMPLE_RESULT_JSON_V1 = 3
  };

  enum CompressionFormat {
    UNKNOWN_COMPRESSION_FORMAT = 0,
    NONE = 1,
    ZLIB_BASE64_V1 = 2
  };
} // namespace cudaq::qio

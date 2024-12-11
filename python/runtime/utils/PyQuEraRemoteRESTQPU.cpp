/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quera/QuEraBaseQPU.h"

namespace cudaq {

class PyQuEraRemoteRESTQPU : public cudaq::QuEraBaseQPU {
public:
  PyQuEraRemoteRESTQPU() : QuEraBaseQPU() {}
  PyQuEraRemoteRESTQPU(PyQuEraRemoteRESTQPU &&) = delete;
  virtual ~PyQuEraRemoteRESTQPU() = default;
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyQuEraRemoteRESTQPU, quera)

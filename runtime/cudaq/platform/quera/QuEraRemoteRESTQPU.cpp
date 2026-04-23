/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"

namespace {

/// @brief The `QuEraRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class QuEraRemoteRESTQPU : public cudaq::AnalogRemoteRESTQPU {
public:
  QuEraRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  QuEraRemoteRESTQPU(QuEraRemoteRESTQPU &&) = delete;
  virtual ~QuEraRemoteRESTQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, QuEraRemoteRESTQPU, quera)

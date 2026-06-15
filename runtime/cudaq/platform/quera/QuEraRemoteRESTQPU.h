/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogRemoteRESTQPU.h"

namespace cudaq {

/// @brief The QuEraRemoteRESTQPU is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Programs via a REST Client.
class QuEraRemoteRESTQPU : public AnalogRemoteRESTQPU {
public:
  QuEraRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  QuEraRemoteRESTQPU(QuEraRemoteRESTQPU &&) = delete;
  ~QuEraRemoteRESTQPU() override;
};

} // namespace cudaq

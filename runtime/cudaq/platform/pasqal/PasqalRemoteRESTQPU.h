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

/// @brief The PasqalRemoteRESTQPU is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Programs via a REST Client.
class PasqalRemoteRESTQPU : public AnalogRemoteRESTQPU {
public:
  PasqalRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  PasqalRemoteRESTQPU(PasqalRemoteRESTQPU &&) = delete;
  ~PasqalRemoteRESTQPU() override;
};

} // namespace cudaq

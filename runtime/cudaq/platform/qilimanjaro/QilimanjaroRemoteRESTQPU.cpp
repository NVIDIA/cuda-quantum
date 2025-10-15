/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"

namespace {

/// @brief The `QilimanjaroRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class QilimanjaroRemoteRESTQPU : public cudaq::AnalogRemoteRESTQPU {
public:
  QilimanjaroRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  QilimanjaroRemoteRESTQPU(QilimanjaroRemoteRESTQPU &&) = delete;
  virtual ~QilimanjaroRemoteRESTQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, QilimanjaroRemoteRESTQPU, qilimanjaro)
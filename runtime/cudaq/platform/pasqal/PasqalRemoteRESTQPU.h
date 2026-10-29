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
protected:
  /// @brief Emulation `C6/hbar` in rad m^6 / s, resolved by `setTargetBackend`.
  double rydbergC6 = 0.0;

  bool supportsEmulation() const override { return true; }
  sample_result emulateJob(const std::string &payload, std::size_t shots,
                           std::size_t seed, analog::Engine &engine) override;

public:
  PasqalRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  void setTargetBackend(const std::string &backend) override;
  PasqalRemoteRESTQPU(PasqalRemoteRESTQPU &&) = delete;
  ~PasqalRemoteRESTQPU() override;
};

} // namespace cudaq

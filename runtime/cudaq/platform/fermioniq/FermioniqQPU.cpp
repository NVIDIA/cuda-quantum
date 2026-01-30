/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqBaseQPU.h"

namespace {

/// @brief The `FermioniqRestQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on the Fermioniq simulator via a REST Client.
class FermioniqRestQPU : public cudaq::FermioniqBaseQPU {
public:
  /// @brief The constructor
  FermioniqRestQPU() : FermioniqBaseQPU() {}

  FermioniqRestQPU(FermioniqRestQPU &&) = delete;
  virtual ~FermioniqRestQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, FermioniqRestQPU, fermioniq)

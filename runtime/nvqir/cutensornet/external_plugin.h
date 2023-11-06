/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cutensornet.h"

namespace nvqir {
/// @brief Interface for externally provided expectation calculation executor
/// (optionally used).
struct CutensornetExecutor {
  virtual std::vector<std::complex<double>>
  computeExpVals(cutensornetHandle_t cutnHandle,
                 cutensornetState_t quantumState, std::size_t numQubits,
                 const std::vector<std::vector<bool>> &symplecticRepr) = 0;
};
} // namespace nvqir

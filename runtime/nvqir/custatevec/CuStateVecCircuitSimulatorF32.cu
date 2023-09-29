/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_CUSTATEVEC_TOGGLE_CREATE
#include "CuStateVecCircuitSimulator.cu"
/// Register this Simulator with NVQIR.
template<>
std::string CuStateVecCircuitSimulator<float>::name() const {
  return "custatevec-fp32";
}
NVQIR_REGISTER_SIMULATOR(CuStateVecCircuitSimulator<float>, custatevec_fp32)

#undef __NVQIR_CUSTATEVEC_TOGGLE_CREATE

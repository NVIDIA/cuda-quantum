/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>

namespace cudaq {
#define CUDAQ_USE_STD20 (__cplusplus >= 202002L)
#define CUDAQ_APPLE_CLANG (defined(__apple_build_version__))

/// @brief Define an enumeration of possible simulation
/// floating point precision types.
enum class simulation_precision { fp32, fp64 };

#ifdef CUDAQ_SIMULATION_SCALAR_FP64
using simulation_scalar = std::complex<double>;
#else
using simulation_scalar = std::complex<float>;
#endif

} // namespace cudaq
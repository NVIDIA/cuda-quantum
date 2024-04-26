/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
<<<<<<< HEAD
#include "target_control.h"

<<<<<<<< HEAD:runtime/cudaq/host_config.h
=======

>>>>>>> upstream/experimental/stateHandling
#pragma once

#include <complex>

#define CUDAQ_USE_STD20 (__cplusplus >= 202002L)
#define CUDAQ_APPLE_CLANG (defined(__apple_build_version__))

namespace cudaq {

/// @brief Define an enumeration of possible simulation
/// floating point precision types.
enum class simulation_precision { fp32, fp64 };

<<<<<<< HEAD
#ifdef CUDAQ_SIMULATION_SCALAR_FP64
using simulation_scalar = std::complex<double>;
#else
using simulation_scalar = std::complex<float>;
#endif

} // namespace cudaq
========
namespace cudaq::__internal__ {
static bool m_CanModifyTarget = true;
void enableTargetModification() { m_CanModifyTarget = true; }
void disableTargetModification() { m_CanModifyTarget = false; }
bool canModifyTarget() { return m_CanModifyTarget; }
} // namespace cudaq::__internal__
>>>>>>>> upstream/experimental/stateHandling:runtime/cudaq/target_control.cpp
=======
#if defined(CUDAQ_SIMULATION_SCALAR_FP64) &&                                   \
    defined(CUDAQ_SIMULATION_SCALAR_FP32)
#error "Simulation precision cannot be both double and float"
#elif defined(CUDAQ_SIMULATION_SCALAR_FP32)
using real = float;
#elif defined(CUDAQ_SIMULATION_SCALAR_FP64)
using real = double;
#else
// If neither precision is specified, assume double.
using real = double;
#endif

using complex = std::complex<real>;

} // namespace cudaq

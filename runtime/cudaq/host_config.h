/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>

#define CUDAQ_APPLE_CLANG (defined(__apple_build_version__))

namespace cudaq {

/// @brief Define an enumeration of possible simulation
/// floating point precision types.
enum class simulation_precision { fp32, fp64 };

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

namespace __internal__ {
/// @brief Static initializer that wires up the runtime target backend on
/// program startup. `nvq++` defines `NVQPP_TARGET_BACKEND_CONFIG` for
/// `gen-target-backend: true` targets so that including any public CUDA-Q
/// header instantiates the inline global below, which calls
/// `quantum_platform::setTargetBackend(...)` from a static constructor.
///
/// The class is a forward declaration so this block has no JIT-compiler
/// dependency; the constructor body lives in `cudaq.cpp`.
class TargetSetter {
public:
  TargetSetter(const char *backend);
};

#ifdef NVQPP_TARGET_BACKEND_CONFIG
inline TargetSetter targetSetter(NVQPP_TARGET_BACKEND_CONFIG);
#endif
} // namespace __internal__

} // namespace cudaq

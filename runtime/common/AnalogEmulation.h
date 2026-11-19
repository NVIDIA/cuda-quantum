/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "common/SampleResult.h"
#include "cudaq/operators.h"
#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/// @brief Vendor-neutral local emulation for analog targets. A target turns its
/// payload into a `Model`; an `Engine`, loaded by name at runtime, evolves it.
namespace cudaq::analog {

/// @brief Time-dependent H/hbar in rad/s with parameter `t` in seconds, one
/// qubit per emulated site, and site 0 at the least significant tensor index.
struct Model {
  sum_op<matrix_handler> hamiltonian;
  dimension_map dimensions;
  /// @brief Waveform breakpoints. Coefficients may be discontinuous at these
  /// times; each interval is integrated with its left limit at the end.
  std::vector<double> times;
  double maxStep;
};

/// @brief Evolves analog models locally. Implementations register by name, e.g.
/// `dynamics` in `libnvqir-dynamics`.
class Engine : public registry::RegisteredType<Engine> {
public:
  virtual ~Engine() = default;
  /// @brief Evolve |0...0> under `model` and return the final state vector.
  virtual std::vector<std::complex<double>>
  evolveFromGround(const Model &model) = 0;
};

/// @brief Return the engine registered as `name`, loading `libnvqir-<name>`
/// from the CUDA-Q library directory if needed. Returns null if it is
/// unavailable.
std::unique_ptr<Engine> loadEngine(const std::string &name);

/// @brief Sample a qubit state vector with site 0 in the least significant
/// index. Output strings list sites in order: vacant or |0>=0, |1>=1. The state
/// vector contains only occupied sites when a filling is supplied.
sample_result sampleStateVector(const std::vector<std::complex<double>> &state,
                                std::size_t shots, std::size_t seed,
                                const std::vector<int> &filling = {});

} // namespace cudaq::analog

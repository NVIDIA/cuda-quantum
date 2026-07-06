/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <curand.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace cudaq::cusv {
/// @brief Generates random numbers on the GPU for state sampling.
class CuStateVecRandom {
public:
  CuStateVecRandom();
  ~CuStateVecRandom();
  CuStateVecRandom(const CuStateVecRandom &) = delete;
  CuStateVecRandom &operator=(const CuStateVecRandom &) = delete;

  void setSeed(std::uint64_t seed);
  std::vector<double> generate(std::size_t count);

private:
  void ensureGenerator();
  void reserve(std::size_t count);

  curandGenerator_t m_generator = nullptr;
  double *m_deviceValues = nullptr;
  std::size_t m_capacity = 0;
  std::optional<std::uint64_t> m_seed;
};

} // namespace cudaq::cusv

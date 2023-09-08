/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_utils.h"
#include "cudaq/platform.h"

#include <random>

namespace cudaq {
std::vector<double> linspace(double a, double b, size_t N) {
  double h = (b - a) / static_cast<double>(N - 1);
  std::vector<double> xs(N);
  typename std::vector<double>::iterator x;
  double val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

std::vector<double> random_vector(const double l_range, const double r_range,
                                  const std::size_t size, const uint32_t seed) {
  // Generate a random initial parameter set
  std::mt19937 mersenne_engine{seed};
  std::uniform_real_distribution<double> dist{l_range, r_range};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<double> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  return vec;
}
} // namespace cudaq

/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.  * All rights reserved.
 *                                                      *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "encoding.h"
#include "cudaq/platform.h"
#include <cmath>
#include <stdexcept>

namespace cudaq::contrib {
namespace {

std::size_t nextPowerOfTwo(std::size_t n) {
  if (n == 0)
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");
  if ((n & (n - 1)) == 0)
    return n;
  std::size_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

simulation_precision targetPrecision() {
  if (const auto *rt = get_platform().get_runtime_target())
    return rt->get_precision();
  return simulation_precision::fp64;
}

void l2NormalizeInPlace(std::vector<std::complex<double>> &vec) {
  long double normSq = 0.0L;
  for (const auto &v : vec)
    normSq += static_cast<long double>(std::norm(v));
  if (normSq == 0.0L)
    throw std::invalid_argument(
        "amplitude_encode: cannot normalize a zero vector.");
  const long double invNorm = 1.0L / std::sqrt(normSq);
  for (auto &v : vec)
    v *= static_cast<double>(invNorm);
}

std::vector<std::complex<double>>
prepareAmplitudeVector(std::span<const std::complex<double>> data,
                       std::complex<double> pad) {
  if (data.empty())
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");

  std::vector<std::complex<double>> vec(data.begin(), data.end());
  const std::size_t targetLen = nextPowerOfTwo(vec.size());
  if (targetLen != vec.size())
    vec.resize(targetLen, pad);
  l2NormalizeInPlace(vec);
  return vec;
}

std::vector<std::complex<double>> stateToAmplitudeVector(const state &data) {
  const auto tensor = data.get_tensor(0);
  const std::size_t numElements = tensor.get_num_elements();
  if (numElements == 0)
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");

  std::vector<std::complex<double>> vec;
  vec.reserve(numElements);
  for (std::size_t i = 0; i < numElements; ++i)
    vec.push_back(data[i]);
  return vec;
}

state fromNormalizedVector(std::vector<std::complex<double>> vec) {
  if (targetPrecision() == simulation_precision::fp32) {
    std::vector<std::complex<float>> fp32(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
      fp32[i] = std::complex<float>(static_cast<float>(vec[i].real()),
                                    static_cast<float>(vec[i].imag()));
    return state::from_data(fp32);
  }
  return state::from_data(vec);
}

} // namespace

state amplitude_encode(std::span<const double> data, std::complex<double> pad) {
  if (data.empty())
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");
  std::vector<std::complex<double>> vec;
  vec.reserve(data.size());
  for (double v : data)
    vec.emplace_back(v, 0.0);
  const std::size_t targetLen = nextPowerOfTwo(vec.size());
  if (targetLen != vec.size())
    vec.resize(targetLen, pad);
  l2NormalizeInPlace(vec);
  return fromNormalizedVector(std::move(vec));
}

state amplitude_encode(std::span<const float> data, std::complex<double> pad) {
  if (data.empty())
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");
  std::vector<double> promoted(data.begin(), data.end());
  return amplitude_encode(std::span<const double>(promoted), pad);
}

state amplitude_encode(std::span<const std::complex<double>> data,
                       std::complex<double> pad) {
  return fromNormalizedVector(prepareAmplitudeVector(data, pad));
}

state amplitude_encode(std::span<const std::complex<float>> data,
                       std::complex<double> pad) {
  if (data.empty())
    throw std::invalid_argument("amplitude_encode: input must be non-empty.");
  std::vector<std::complex<double>> promoted;
  promoted.reserve(data.size());
  for (const auto &v : data)
    promoted.emplace_back(v.real(), v.imag());
  return fromNormalizedVector(prepareAmplitudeVector(
      std::span<const std::complex<double>>(promoted), pad));
}

state amplitude_encode(const state &data, std::complex<double> pad) {
  auto vec = stateToAmplitudeVector(data);
  const std::size_t targetLen = nextPowerOfTwo(vec.size());
  if (targetLen != vec.size())
    vec.resize(targetLen, pad);
  l2NormalizeInPlace(vec);
  return fromNormalizedVector(std::move(vec));
}

} // namespace cudaq::contrib

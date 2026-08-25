/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecRandom.h"

#include "CuStateVecError.h"

#include <cmath>
#include <random>

namespace cudaq::cusv {

CuStateVecRandom::CuStateVecRandom() = default;

void CuStateVecRandom::ensureGenerator() {
  if (m_generator)
    return;
  HANDLE_CURAND_ERROR(
      curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT));
  const auto seed = m_seed.value_or(std::random_device{}());
  HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(m_generator, seed));
  HANDLE_CURAND_ERROR(curandSetGeneratorOffset(m_generator, 0));
}

CuStateVecRandom::~CuStateVecRandom() {
  if (m_deviceValues)
    cudaFree(m_deviceValues);
  if (m_generator)
    curandDestroyGenerator(m_generator);
}

void CuStateVecRandom::setSeed(std::uint64_t seed) {
  m_seed = seed;
  ensureGenerator();
  HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(m_generator, seed));
  HANDLE_CURAND_ERROR(curandSetGeneratorOffset(m_generator, 0));
}

void CuStateVecRandom::reserve(std::size_t count) {
  if (count <= m_capacity)
    return;
  if (m_deviceValues)
    HANDLE_CUDA_ERROR(cudaFree(m_deviceValues));
  m_deviceValues = nullptr;
  m_capacity = 0;
  HANDLE_CUDA_ERROR(cudaMalloc(&m_deviceValues, count * sizeof(double)));
  m_capacity = count;
}

std::vector<double> CuStateVecRandom::generate(std::size_t count) {
  if (count == 0)
    return {};
  ensureGenerator();
  reserve(count);
  HANDLE_CURAND_ERROR(
      curandGenerateUniformDouble(m_generator, m_deviceValues, count));
  std::vector<double> result(count);
  HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), m_deviceValues,
                               count * sizeof(double), cudaMemcpyDeviceToHost));
  for (double &value : result)
    if (value == 1.0)
      value = std::nextafter(1.0, 0.0);
  return result;
}

} // namespace cudaq::cusv

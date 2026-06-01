/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

extern "C" __device__ int addThem(int a, int b) { return a + b; }

extern "C" __device__ float multiplyFloats(float a, float b) {
  return a * b;
}

extern "C" __device__ int
countTrueBits(const bool *bits, std::uint64_t count, std::uint64_t bias) {
  int total = static_cast<int>(bias);
  for (std::uint64_t i = 0; i < count; ++i)
    total += bits[i] ? 1 : 0;
  return total;
}

extern "C" __device__ int
countTrueMeasures(const bool *bits, std::uint64_t count, std::uint64_t bias) {
  return countTrueBits(bits, count, bias);
}

extern "C" __device__ int
sumIntVector(const int *values, std::uint64_t count, int bias) {
  int total = bias;
  for (std::uint64_t i = 0; i < count; ++i)
    total += values[i];
  return total;
}

extern "C" __device__ void
incrementIntVector(int *out, std::uint64_t outCount, const int *values,
                   std::uint64_t count, int delta) {
  std::uint64_t limit = outCount < count ? outCount : count;
  for (std::uint64_t i = 0; i < limit; ++i)
    out[i] = values[i] + delta;
}

extern "C" __device__ void
integerToBinaryVector(bool *out, std::uint64_t outCount, std::uint64_t value) {
  for (std::uint64_t i = 0; i < outCount; ++i)
    out[i] = ((value >> i) & 1u) != 0;
}

extern "C" __device__ float
sumFloatVector(const float *values, std::uint64_t count, float bias) {
  float total = bias;
  for (std::uint64_t i = 0; i < count; ++i)
    total += values[i];
  return total;
}

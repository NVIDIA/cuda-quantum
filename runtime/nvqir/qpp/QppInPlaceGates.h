/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The routines in this file are a qubit-only, in-place replacement for the
// `qpp::apply` / `qpp::applyCTRL` routines of Quantum++
// (https://github.com/softwareqinc/qpp), which is distributed under the
// following license:
//
// Copyright (c) 2013 - 2022 softwareQ Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace nvqir {
namespace details {

/// @brief Number of threads to use for a loop over a state vector of the
/// given dimension. Small states run serially, since waking the thread pool
/// costs more than the loop, and mid-sized ones use part of the pool. The
/// count depends only on the dimension so that every loop of one execution
/// uses the same team: resizing the team between loops is expensive.
inline int parallelThreads(std::size_t dimension) {
#if defined(_OPENMP)
  const auto maxThreads =
      static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
  if (dimension < (1ULL << 15))
    return 1;
  return static_cast<int>(
      std::clamp<std::size_t>(dimension >> 13, 1, maxThreads));
#else
  return 1;
#endif
}

/// @brief Insert a 0 bit at each of the given (ascending) bit positions.
inline std::size_t insertZeroBits(std::size_t value,
                                  const std::vector<std::size_t> &sortedBits) {
  for (auto bit : sortedBits) {
    const std::size_t low = value & ((1ULL << bit) - 1);
    value = ((value >> bit) << (bit + 1)) | low;
  }
  return value;
}

/// @brief Complex product without the NaN/Inf handling of `operator*`, which
/// is much slower in the hot loops.
inline std::complex<double> multiply(const std::complex<double> &a,
                                     const std::complex<double> &b) {
  return {a.real() * b.real() - a.imag() * b.imag(),
          a.real() * b.imag() + a.imag() * b.real()};
}

/// @brief Apply a (controlled) gate to a state vector, in place.
///
/// Qubit `q` is bit `q` of the amplitude index. `matrix` is row-major, with
/// `targets[0]` as the most significant bit of its index. Only amplitudes
/// whose control bits are all 1 are touched.
inline void applyGateInPlace(std::complex<double> *state, std::size_t numQubits,
                             const std::vector<std::complex<double>> &matrix,
                             const std::vector<std::size_t> &controls,
                             const std::vector<std::size_t> &targets) {
  const std::size_t numTargets = targets.size();
  if (numTargets == 0)
    throw std::invalid_argument("[qpp] gate applied to no target qubits.");
  if (numTargets >= numQubits + 1)
    throw std::invalid_argument("[qpp] gate has more targets than qubits.");
  const std::size_t dim = 1ULL << numTargets;
  if (matrix.size() != dim * dim)
    throw std::invalid_argument(
        "[qpp] gate matrix size does not match the number of targets.");

  // Collect and validate the control and target qubits.
  std::size_t controlMask = 0, involvedMask = 0;
  std::vector<std::size_t> involved;
  involved.reserve(controls.size() + numTargets);
  const auto addQubit = [&](std::size_t qubit) {
    if (qubit >= numQubits)
      throw std::invalid_argument("[qpp] qubit index out of range.");
    const std::size_t bit = 1ULL << qubit;
    if (involvedMask & bit)
      throw std::invalid_argument(
          "[qpp] control and target qubits must be distinct.");
    involvedMask |= bit;
    involved.push_back(qubit);
  };
  for (auto qubit : controls) {
    addQubit(qubit);
    controlMask |= 1ULL << qubit;
  }
  for (auto qubit : targets)
    addQubit(qubit);
  std::sort(involved.begin(), involved.end());

  // One group of amplitudes per assignment of the remaining qubits.
  const std::size_t numGroups = 1ULL << (numQubits - involved.size());
  [[maybe_unused]] const int numThreads = parallelThreads(1ULL << numQubits);

  if (numTargets == 1) {
    const std::size_t targetBit = 1ULL << targets[0];
    const auto m00 = matrix[0], m01 = matrix[1], m10 = matrix[2],
               m11 = matrix[3];
#if defined(_OPENMP)
#pragma omp parallel for num_threads(numThreads) if (numThreads > 1)
#endif
    for (std::size_t i = 0; i < numGroups; ++i) {
      const std::size_t i0 = insertZeroBits(i, involved) | controlMask;
      const std::size_t i1 = i0 | targetBit;
      const auto a0 = state[i0], a1 = state[i1];
      state[i0] = multiply(m00, a0) + multiply(m01, a1);
      state[i1] = multiply(m10, a0) + multiply(m11, a1);
    }
    return;
  }

  // Index offset of each target basis state within a group.
  std::vector<std::size_t> offsets(dim, 0);
  for (std::size_t m = 0; m < dim; ++m)
    for (std::size_t j = 0; j < numTargets; ++j)
      if (m & (1ULL << (numTargets - 1 - j)))
        offsets[m] |= 1ULL << targets[j];

#if defined(_OPENMP)
#pragma omp parallel num_threads(numThreads) if (numThreads > 1)
#endif
  {
    std::vector<std::complex<double>> amplitudes(dim);
#if defined(_OPENMP)
#pragma omp for
#endif
    for (std::size_t i = 0; i < numGroups; ++i) {
      const std::size_t base = insertZeroBits(i, involved) | controlMask;
      for (std::size_t m = 0; m < dim; ++m)
        amplitudes[m] = state[base | offsets[m]];
      for (std::size_t row = 0; row < dim; ++row) {
        std::complex<double> sum = 0.0;
        const auto *matrixRow = matrix.data() + row * dim;
        for (std::size_t col = 0; col < dim; ++col)
          sum += multiply(matrixRow[col], amplitudes[col]);
        state[base | offsets[row]] = sum;
      }
    }
  }
}

} // namespace details

} // namespace nvqir

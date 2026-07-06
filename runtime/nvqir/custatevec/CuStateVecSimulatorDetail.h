/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecConfig.h"
#include "CuStateVecDevice.h"
#include "CuStateVecError.h"
#include "CuStateVecTasks.h"
#include "common/ObserveResult.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudaq::cusv::detail {

/// Translate a CUDA-Q Pauli operator into its cuStateVec enumerator.
inline custatevecPauli_t toCuStateVecPauli(cudaq::pauli value) {
  switch (value) {
  case cudaq::pauli::I:
    return CUSTATEVEC_PAULI_I;
  case cudaq::pauli::X:
    return CUSTATEVEC_PAULI_X;
  case cudaq::pauli::Y:
    return CUSTATEVEC_PAULI_Y;
  case cudaq::pauli::Z:
    return CUSTATEVEC_PAULI_Z;
  }
  throw std::invalid_argument("Invalid Pauli operator.");
}

/// Copy `size` state-vector amplitudes from a host or device pointer into a
/// host vector, dispatching on the pointer's residence.
template <typename Scalar>
std::vector<std::complex<Scalar>> copyPointerToHost(const void *data,
                                                    std::size_t size) {
  if (!data)
    throw std::invalid_argument("State-vector data pointer cannot be null.");
  std::vector<std::complex<Scalar>> result(size);
  cudaPointerAttributes attributes{};
  const cudaError_t status =
      cudaPointerGetAttributes(&attributes, const_cast<void *>(data));
  if (status == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
    HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), data,
                                 result.size() * sizeof(result.front()),
                                 cudaMemcpyDeviceToHost));
  } else {
    if (status != cudaSuccess)
      cudaGetLastError();
    std::copy_n(static_cast<const std::complex<Scalar> *>(data), size,
                result.begin());
  }
  return result;
}

/// Narrow a vector of qubit indices to the int32 range cuStateVec expects,
/// rejecting indices that do not fit.
inline std::vector<int32_t> toInt32(const std::vector<std::size_t> &values) {
  std::vector<int32_t> result;
  result.reserve(values.size());
  for (const std::size_t value : values) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
      throw std::out_of_range("Qubit index exceeds cuStateVec range.");
    result.push_back(static_cast<int32_t>(value));
  }
  return result;
}

/// Restores a Boolean flag to its previous value when leaving a scope.
class ScopedFlag {
public:
  explicit ScopedFlag(bool &flag) : m_flag(flag), m_previous(flag) {
    m_flag = true;
  }
  ~ScopedFlag() { m_flag = m_previous; }

private:
  bool &m_flag;
  bool m_previous;
};

/// Build the simulator configuration from the environment and the current
/// device's compute capability.
template <typename Scalar>
CuStateVecConfig deviceConfig() {
  const auto capacity = queryDeviceMemoryCapacity();
  return CuStateVecConfig::fromEnvironment(capacity.computeCapabilityMajor,
                                           capacity.computeCapabilityMinor,
                                           std::is_same_v<Scalar, float>);
}

/// Reduce a histogram of measured bit-strings to a Z-parity expectation:
/// even-parity outcomes add +probability, odd-parity outcomes add -probability.
inline double parityExpectation(const cudaq::ExecutionResult &result,
                                int shots) {
  assert(shots > 0 && "Parity expectation requires a positive shot count.");
  double expectation = 0.0;
  for (const auto &[bits, count] : result.counts) {
    const double probability = static_cast<double>(count) / shots;
    expectation += cudaq::sample_result::has_even_parity(bits) ? probability
                                                               : -probability;
  }
  return expectation;
}

/// The non-identity Pauli factors of each spin-op term, in term-iteration
/// order.
struct PauliTerms {
  std::vector<std::vector<custatevecPauli_t>> paulis;
  std::vector<std::vector<int32_t>> targets;
};

/// Extract, per term of `op`, the non-identity Pauli operators and their target
/// wires. Identity factors are dropped because cuStateVecEx encodes them
/// implicitly through the absence of a wire.
inline PauliTerms extractPauliTerms(const cudaq::spin_op &op) {
  PauliTerms terms;
  terms.paulis.reserve(op.num_terms());
  terms.targets.reserve(op.num_terms());
  for (const auto &term : op) {
    auto &paulis = terms.paulis.emplace_back();
    auto &targets = terms.targets.emplace_back();
    for (const auto &termOp : term) {
      if (termOp.as_pauli() == cudaq::pauli::I)
        continue;
      paulis.push_back(toCuStateVecPauli(termOp.as_pauli()));
      targets.push_back(static_cast<int32_t>(termOp.target()));
    }
  }
  assert(terms.paulis.size() == op.num_terms() &&
         "Expected one Pauli group per spin-op term.");
  return terms;
}

/// Assemble an observe_result from per-term expectation sums accumulated over
/// `trajectoryCount` trajectories. `averaged[i]` holds the summed expectation
/// of term `i` of `op`, in term-iteration order; it is normalized here.
inline cudaq::observe_result makeObserveResult(const cudaq::spin_op &op,
                                               std::vector<double> averaged,
                                               std::size_t trajectoryCount) {
  assert(averaged.size() == op.num_terms() &&
         "One accumulated expectation is required per spin-op term.");
  assert(trajectoryCount > 0 &&
         "Trajectory averaging requires a positive trajectory count.");
  double expectation = 0.0;
  std::vector<cudaq::ExecutionResult> termResults;
  termResults.reserve(op.num_terms());
  std::size_t termIndex = 0;
  for (const auto &term : op) {
    averaged[termIndex] /= static_cast<double>(trajectoryCount);
    expectation += (term.evaluate_coefficient() * averaged[termIndex]).real();
    termResults.emplace_back(
        cudaq::ExecutionResult({}, term.get_term_id(), averaged[termIndex]));
    ++termIndex;
  }
  return cudaq::observe_result(
      expectation, op,
      cudaq::sample_result(expectation, std::move(termResults)));
}

} // namespace cudaq::cusv::detail

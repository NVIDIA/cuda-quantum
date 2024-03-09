/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "kernel_builder.h"
#include <complex>

namespace cudaq {

namespace details {

/// @brief Converts angles of a uniformly controlled rotation to angles of
/// non-controlled rotations.
std::vector<double> convertAngles(const std::span<double> alphas);

/// @brief Return the control indices dictated by the gray code implementation.
///
/// Here, numBits is the number of controls.
std::vector<std::size_t> getControlIndices(std::size_t numBits);

/// @brief Apply a uniformly controlled rotation on the target qubit.
template <typename Kernel, typename RotationFunctor>
void applyRotation(Kernel &&kernel, RotationFunctor &&rotationFunctor,
                   QuakeValue qubits, const std::span<double> alphas,
                   std::size_t numControls, std::size_t target) {
  auto thetas = convertAngles(alphas);
  if (numControls == 0) {
    rotationFunctor(kernel, thetas[0], qubits[target]);
    return;
  }

  auto controlIndices = getControlIndices(numControls);
  assert(thetas.size() == controlIndices.size());
  for (auto [i, c] : cudaq::enumerate(controlIndices)) {
    rotationFunctor(kernel, thetas[i], qubits[target]);
    kernel.template x<cudaq::ctrl>(qubits[c], qubits[target]);
  }
}

/// @brief Return angles required to implement a uniformly controlled z-rotation
/// on the `kth` qubit.
std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);

/// @brief Return angles required to implement a uniformly controlled y-rotation
/// on the `kth` qubit.
std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);
} // namespace details

/// @brief Decompose the input state vector data to a set of controlled
/// operations and rotations. This function takes as input a `kernel_builder`
/// and appends the operations of the decomposition to its internal
/// representation. This implementation follows the algorithm defined in
/// `https://arxiv.org/pdf/quant-ph/0407010.pdf`.
template <typename Kernel>
void from_state(Kernel &&kernel, QuakeValue &qubits,
                const std::span<std::complex<double>> amplitudes,
                std::size_t inNumQubits = 0) {
  auto numQubits = qubits.constantSize().value_or(inNumQubits);
  if (numQubits == 0)
    throw std::runtime_error(
        "[from_state] cannot infer size of input quantum register, please "
        "specify the number of qubits via the from_state() final argument.");
  if ((1ULL << numQubits) != amplitudes.size())
    throw std::runtime_error(
        std::string("[from_state] mismatch between number of qubits, n = ") +
        std::to_string(numQubits) + ", and state dimension, " +
        std::to_string(amplitudes.size()) +
        ". The dimension of the state must be 2 ** n");

  // Decompose the state into phases and magnitudes.
  bool needsPhaseEqualization = false;
  std::vector<double> phases;
  std::vector<double> magnitudes;
  for (const auto &a : amplitudes) {
    phases.push_back(std::arg(a));
    magnitudes.push_back(std::abs(a));
    // FIXME: remove magic number.
    needsPhaseEqualization |= std::abs(phases.back()) > 1e-10;
  }

  // N.B: The algorithm, as described in the paper, creates a circuit that
  // begins with a target state and brings it to the all zero state. Hence, this
  // implementation do the two steps described in Section III in reverse order.
  auto applyRy = [](auto &&kernel, auto &&theta, auto &&qubit) {
    kernel.ry(theta, qubit);
  };
  // Apply uniformly controlled y-rotations, the constrution in Eq. (4).
  for (std::size_t j = 1; j <= numQubits; ++j) {
    auto k = numQubits - j + 1;
    auto numControls = j - 1;
    auto target = j - 1;
    auto alphaYk = details::getAlphaY(magnitudes, numQubits, k);
    details::applyRotation(kernel, applyRy, qubits, alphaYk, numControls,
                           target);
  }

  if (!needsPhaseEqualization)
    return;

  // Apply uniformly controlled z-rotations, the constrution in Eq. (4).
  auto applyRz = [](auto &&kernel, auto &&theta, auto &&qubit) {
    kernel.rz(theta, qubit);
  };
  for (std::size_t j = 1; j <= numQubits; ++j) {
    auto k = numQubits - j + 1;
    auto numControls = j - 1;
    auto target = j - 1;
    auto alphaZk = details::getAlphaZ(phases, numQubits, k);
    if (alphaZk.empty())
      continue;
    details::applyRotation(kernel, applyRz, qubits, alphaZk, numControls,
                           target);
  }
}

/// @brief Construct a CUDA Quantum kernel that produces the given state. This
/// overload will return the `kernel_builder` as a `unique_ptr`.
inline auto from_state(const std::span<std::complex<double>> data) {
  auto numQubits = std::log2(data.size());
  std::vector<details::KernelBuilderType> empty;
  auto kernel = std::make_unique<kernel_builder<>>(empty);
  auto qubits = kernel->qalloc(numQubits);
  from_state(*kernel.get(), qubits, data);
  return kernel;
}

} // namespace cudaq

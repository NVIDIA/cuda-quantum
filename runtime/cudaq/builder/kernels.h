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

/// @brief Convert the provided angles to those rotation angles
/// used for the gray code implementation.
std::vector<double> computeAngle(const std::vector<double> &alpha);

/// @brief Return the control indices dictated by the gray code implementation.
std::vector<std::size_t> getControlIndices(std::size_t grayRank);

/// @brief Apply a uniformly controlled rotation on the target qubit.
template <typename Kernel, typename RotationFunctor>
void applyRotation(Kernel &&kernel, RotationFunctor &&rotationFunctor,
                   QuakeValue qubits, const std::vector<double> &alpha,
                   const std::vector<std::size_t> &controls,
                   std::size_t target) {
  auto thetas = computeAngle(alpha);
  auto gcRank = controls.size();
  if (gcRank == 0) {
    rotationFunctor(kernel, thetas[0], qubits[target]);
    return;
  }

  auto ctrlIds = details::getControlIndices(gcRank);
  for (auto [i, ctrlIdx] : cudaq::enumerate(ctrlIds)) {
    rotationFunctor(kernel, thetas[i], qubits[target]);
    kernel.template x<cudaq::ctrl>(qubits[controls[ctrlIdx]], qubits[target]);
  }
}

/// @brief Return angles required to implement a controlled-Z rotation on
/// the `kth` qubit.
std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);

/// @brief Return angles required to implement a controlled-Y rotation on
/// the `kth` qubit.
std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);
} // namespace details

/// @brief Decompose the input state vector data to a set of
/// controlled operations and rotations. This function takes as input
/// a `kernel_builder` and appends the operations of the decomposition
/// to its internal representation. This implementation follows the algorithm
/// defined in `https://arxiv.org/pdf/quant-ph/0407010.pdf`.
template <typename Kernel>
void from_state(Kernel &&kernel, QuakeValue &qubits,
                const std::span<std::complex<double>> data,
                std::size_t inNumQubits = 0) {
  auto numQubits = qubits.constantSize().value_or(inNumQubits);
  if (numQubits == 0)
    throw std::runtime_error(
        "[from_state] cannot infer size of input quantum register, please "
        "specify the number of qubits via the from_state() final argument.");

  auto mutableQubits = cudaq::range(numQubits);
  std::reverse(mutableQubits.begin(), mutableQubits.end());
  bool omegaNonZero = false;
  std::vector<double> omega, stateAbs;
  for (auto &d : data) {
    omega.push_back(std::arg(d));
    stateAbs.push_back(std::abs(d));
    if (std::fabs(omega.back()) > 1e-6)
      omegaNonZero = true;
  }

  for (std::size_t k = mutableQubits.size(); k > 0; k--) {
    auto alphaYk = details::getAlphaY(stateAbs, mutableQubits.size(), k);
    std::vector<std::size_t> controls(mutableQubits.begin() + k,
                                      mutableQubits.end());
    auto target = mutableQubits[k - 1];
    details::applyRotation(
        kernel,
        [](auto &&kernel, auto &&theta, auto &&qubit) {
          kernel.ry(theta, qubit);
        },
        qubits, alphaYk, controls, target);
  }

  if (omegaNonZero) {
    for (std::size_t k = mutableQubits.size(); k > 0; k--) {
      auto alphaZk = details::getAlphaZ(omega, mutableQubits.size(), k);
      std::vector<std::size_t> controls(mutableQubits.begin() + k,
                                        mutableQubits.end());
      auto target = mutableQubits[k - 1];
      if (!alphaZk.empty())
        details::applyRotation(
            kernel,
            [](auto &&kernel, auto &&theta, auto &&qubit) {
              kernel.rz(theta, qubit);
            },
            qubits, alphaZk, controls, target);
    }
  }
}

/// @brief Construct a CUDA Quantum kernel that produces the
/// given state. This overload will return the `kernel_builder` as a
/// `unique_ptr`.
auto from_state(const std::span<std::complex<double>> data) {
  auto numQubits = std::log2(data.size());
  std::vector<details::KernelBuilderType> empty;
  auto kernel = std::make_unique<kernel_builder<>>(empty);
  auto qubits = kernel->qalloc(numQubits);
  from_state(*kernel.get(), qubits, data);
  return kernel;
}

} // namespace cudaq

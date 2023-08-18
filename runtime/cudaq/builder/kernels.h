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

std::vector<std::string> grayCode(std::size_t rank);
std::size_t mEntry(std::size_t row, std::size_t col);
std::vector<double> computeAngle(std::vector<double> alpha);
std::vector<std::size_t> getControlIndicesFromGrayCode(std::size_t grayRank);

template <typename Kernel, typename RotationFunctor>
void applyRotation(Kernel &&kernel, RotationFunctor &&rotationFunctor,
                   QuakeValue qubits, std::vector<double> alpha,
                   std::vector<std::size_t> controls, std::size_t target) {
  auto thetas = computeAngle(alpha);
  auto gcRank = controls.size();
  if (gcRank == 0) {
    rotationFunctor(kernel, thetas[0], qubits[target]);
    return;
  }

  auto ctrlIds = details::getControlIndicesFromGrayCode(gcRank);
  for (auto [i, ctrlIdx] : cudaq::enumerate(ctrlIds)) {
    rotationFunctor(kernel, thetas[i], qubits[target]);
    kernel.template x<cudaq::ctrl>(qubits[controls[ctrlIdx]], qubits[target]);
  }
}
std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);
std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);
} // namespace details

template <typename Kernel>
void from_state(Kernel &&kernel, QuakeValue &qubits,
                const std::span<std::complex<double>> data,
                const std::vector<std::size_t> &wires) {
  bool omegaNonZero = false;
  std::vector<double> omega, stateAbs;
  for (auto &d : data) {
    omega.push_back(std::arg(d));
    stateAbs.push_back(std::fabs(d));
    if (std::fabs(omega.back()) > 1e-6)
      omegaNonZero = true;
  }

  for (std::size_t k = wires.size(); k > 0; k--) {
    auto alphaYk = details::getAlphaY(stateAbs, wires.size(), k);
    std::vector<std::size_t> controls(wires.begin() + k, wires.end());
    auto target = wires[k - 1];
    details::applyRotation(
        kernel,
        [](auto &&kernel, auto &&theta, auto &&qubit) {
          kernel.ry(theta, qubit);
        },
        qubits, alphaYk, controls, target);
  }

  if (omegaNonZero) {
    for (std::size_t k = wires.size(); k > 0; k--) {
      auto alphaZk = details::getAlphaZ(omega, wires.size(), k);
      std::vector<std::size_t> controls(wires.begin() + k, wires.end());
      auto target = wires[k - 1];
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
} // namespace cudaq
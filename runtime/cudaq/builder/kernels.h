/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "kernel_builder.h"
#include <complex>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>

namespace cudaq {

namespace detail {

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

  auto ctrlIds = detail::getControlIndices(gcRank);
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

} // namespace detail

/// @brief Decompose the input state vector data to a set of
/// controlled operations and rotations. This function takes as input
/// a `kernel_builder` and appends the operations of the decomposition
/// to its internal representation. This implementation follows the algorithm
/// defined in `https://arxiv.org/pdf/quant-ph/0407010.pdf`.
template <typename Kernel>
void from_state(Kernel &&kernel, QuakeValue &qubits,
                const std::span<std::complex<double>> data,
                std::size_t inNumQubits = 0) {
  std::make_signed_t<std::size_t> numQubits =
      qubits.constantSize().value_or(inNumQubits);
  if (numQubits <= 0)
    throw std::runtime_error(
        "[from_state] cannot infer size of input quantum register, please "
        "specify the number of qubits via the from_state() final argument.");

  constexpr double basisTol = 1e-12;
  std::size_t nonZeroCount = 0;
  std::size_t nonZeroIdx = 0;
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (std::abs(data[i]) > basisTol) {
      ++nonZeroCount;
      nonZeroIdx = i;
      if (nonZeroCount > 1)
        break;
    }
  }
  if (nonZeroCount == 0)
    throw std::invalid_argument(
        "[from_state] input state vector is all zeros; a quantum state "
        "must have unit norm.");
  if (nonZeroCount == 1) {
    // Möttönen ordering: state-vector index MSB maps to qubits[0], LSB to
    // qubits[numQubits-1].
    auto nq = static_cast<std::size_t>(numQubits);
    for (std::size_t q = 0; q < nq; ++q)
      if ((nonZeroIdx >> (nq - 1 - q)) & 1)
        kernel.x(qubits[q]);
    return;
  }

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
    auto alphaYk = detail::getAlphaY(stateAbs, mutableQubits.size(), k);
    std::vector<std::size_t> controls(mutableQubits.begin() + k,
                                      mutableQubits.end());
    auto target = mutableQubits[k - 1];
    detail::applyRotation(
        kernel,
        [](auto &&kernel, auto &&theta, auto &&qubit) {
          kernel.ry(theta, qubit);
        },
        qubits, alphaYk, controls, target);
  }

  if (omegaNonZero) {
    for (std::size_t k = mutableQubits.size(); k > 0; k--) {
      auto alphaZk = detail::getAlphaZ(omega, mutableQubits.size(), k);
      std::vector<std::size_t> controls(mutableQubits.begin() + k,
                                        mutableQubits.end());
      auto target = mutableQubits[k - 1];
      if (!alphaZk.empty())
        detail::applyRotation(
            kernel,
            [](auto &&kernel, auto &&theta, auto &&qubit) {
              kernel.rz(theta, qubit);
            },
            qubits, alphaZk, controls, target);
    }
  }
}

/// @brief Construct a CUDA-Q kernel that produces the
/// given state. This overload will return the `kernel_builder` as a
/// `unique_ptr`.
auto from_state(const std::span<std::complex<double>> data) {
  auto numQubits = std::log2(data.size());
  std::vector<detail::KernelBuilderType> empty;
  auto kernel = std::make_unique<kernel_builder<>>(empty);
  auto qubits = kernel->qalloc(numQubits);
  from_state(*kernel.get(), qubits, data);
  return kernel;
}

namespace contrib {

/// @brief Pauli axis for ``cudaq::contrib::angular_encode`` rotations
/// \f$R_P(\theta) = e^{-i \theta P / 2}\f$.
enum class RotationAxis { X, Y, Z };

namespace detail {

template <typename Kernel>
void applyAxisRotation(Kernel &kernel, RotationAxis axis, QuakeValue theta,
                       QuakeValue qubit) {
  switch (axis) {
  case RotationAxis::X:
    kernel.rx(theta, qubit);
    break;
  case RotationAxis::Y:
    kernel.ry(theta, qubit);
    break;
  case RotationAxis::Z:
    kernel.rz(theta, qubit);
    break;
  }
}

template <typename Kernel>
void applyAxisRotation(Kernel &kernel, RotationAxis axis, double theta,
                       QuakeValue qubit) {
  switch (axis) {
  case RotationAxis::X:
    kernel.rx(theta, qubit);
    break;
  case RotationAxis::Y:
    kernel.ry(theta, qubit);
    break;
  case RotationAxis::Z:
    kernel.rz(theta, qubit);
    break;
  }
}

inline void validateAngularEncodeSizes(std::optional<std::size_t> qSize,
                                       std::size_t numAngles) {
  if (qSize && *qSize != numAngles)
    throw std::runtime_error(
        "cudaq.contrib.angular_encode: number of angles must match the "
        "number of qubits");
}

} // namespace detail

/// @brief Encode classical features as single-qubit rotation gates.
///
/// Angular (rotation) encoding maps a classical angle vector
/// \f$\boldsymbol{\theta} = (\theta_0, \ldots, \theta_{n-1})\f$ to an
/// \f$n\f$-qubit product state by applying one parameterized rotation per
/// qubit. Starting from \f$|0\rangle^{\otimes n}\f$, the encoded state is
/// \f[
///   |\psi\rangle
///   = \bigotimes_{i=0}^{n-1} R_{\mathrm{axis}}(\theta_i)\,|0\rangle
///   = \prod_{i=0}^{n-1} R_{\mathrm{axis}}(\theta_i)\,|0\rangle_i,
/// \f]
/// where the product applies \f$R_{\mathrm{axis}}(\theta_i)\f$ on qubit \f$i\f$
/// and leaves other qubits unchanged. CUDA-Q uses
/// \f$R_P(\theta) = e^{-i \theta P / 2}\f$ for \f$P \in \{X, Y, Z\}\f$,
/// implemented as ``rx``, ``ry``, or ``rz`` for ``RotationAxis::X``, ``Y``,
/// or ``Z`` respectively. For example, ``RotationAxis::Y`` gives
/// \f$R_Y(\theta_i)|0\rangle = \cos(\theta_i/2)|0\rangle +
/// \sin(\theta_i/2)|1\rangle\f$ on qubit \f$i\f$.
///
/// For use with ``cudaq::kernel_builder`` only (not ``__qpu__`` kernels).
///
/// @param kernel The kernel builder recording the circuit.
/// @param q A ``qvector`` register to encode into.
/// @param angles Rotation angles \f$\theta_i\f$ as a ``std::vector<double>``
/// kernel argument.
/// @param rotation Rotation axis (default ``RotationAxis::Y``).
template <typename Kernel>
void angular_encode(Kernel &&kernel, QuakeValue &q, QuakeValue &angles,
                    RotationAxis rotation = RotationAxis::Y) {
  if (!angles.isStdVec())
    throw std::runtime_error(
        "cudaq.contrib.angular_encode: angles must be std::vector<double>");

  std::function<void(QuakeValue &)> body = [&](QuakeValue &i) {
    detail::applyAxisRotation(kernel, rotation, angles[i], q[i]);
  };
  kernel.for_loop(0, q.size(), std::move(body));
}

/// @copydoc angular_encode
///
/// Overload for a fixed list of angles known at kernel-build time. When
/// ``q`` has a compile-time size, the length of ``angles`` must match the
/// number of qubits.
template <typename Kernel>
void angular_encode(Kernel &&kernel, QuakeValue &q,
                    std::span<const double> angles,
                    RotationAxis rotation = RotationAxis::Y) {
  detail::validateAngularEncodeSizes(q.constantSize(), angles.size());
  for (std::size_t i = 0; i < angles.size(); ++i)
    detail::applyAxisRotation(kernel, rotation, angles[i], q[i]);
}

/// @copydoc angular_encode
template <typename Kernel>
void angular_encode(Kernel &&kernel, QuakeValue &q,
                    const std::vector<double> &angles,
                    RotationAxis rotation = RotationAxis::Y) {
  angular_encode(kernel, q, std::span<const double>(angles), rotation);
}

} // namespace contrib

} // namespace cudaq

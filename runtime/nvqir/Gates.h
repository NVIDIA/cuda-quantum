/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
#include <complex>
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif

#include <vector>

namespace nvqir {

using namespace std::complex_literals;

template <typename Scalar = double>
static constexpr std::complex<Scalar> im = std::complex<Scalar>(0, 1.);

template <typename ScalarType = double>
using ComplexT = std::complex<ScalarType>;

/// @brief Enumeration of supported CUDA Quantum operations
enum class GateName {
  X,
  Y,
  Z,
  H,
  S,
  Sdg,
  Tdg,
  T,
  Rx,
  Ry,
  Rz,
  R1,
  U1,
  U2,
  U3,
  PhasedRx
};

/// @brief Given the gate name (an element of the GateName enum),
/// return the matrix data, optionally parameterized by a rotation angle.
template <typename Scalar>
std::vector<std::complex<Scalar>>
getGateByName(GateName name, const std::vector<Scalar> angles = {}) {
  Scalar two = 2.;
  switch (name) {
  case (GateName::X):
    return {{0., 0.}, {1.0, 0.}, {1.0, 0.0}, {0., 0.}};
  case (GateName::Y):
    return {{0., 0.}, {0.0, -1.0}, {0.0, 1.0}, {0., 0.}};
  case (GateName::Z):
    return {{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {-1., 0.}};
  case (GateName::H): {
    Scalar oneOverSqrt2 = 1 / std::sqrt(2.);
    return {oneOverSqrt2, oneOverSqrt2, oneOverSqrt2, -oneOverSqrt2};
  }
  case (GateName::S):
    return {{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {0., 1.}};
  case (GateName::Sdg):
    return {{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {0., -1.}};
  case (GateName::T):
    return {{1., 0.},
            {0.0, 0.},
            {0.0, 0.0},
            std::exp(im<Scalar> * static_cast<Scalar>(M_PI_4))};
  case (GateName::Tdg):
    return {{1., 0.},
            {0.0, 0.},
            {0.0, 0.0},
            std::exp(-im<Scalar> * static_cast<Scalar>(M_PI_4))};
  case (GateName::Rx): {
    auto angle = angles[0];
    return {{std::cos(angle / two), 0.},
            {0., -1 * std::sin(angle / two)},
            {0, -1 * std::sin(angle / two)},
            {std::cos(angle / two), 0.}};
  }
  case (GateName::Ry): {
    auto angle = angles[0];
    return {std::cos(angle / two), -std::sin(angle / two),
            std::sin(angle / two), std::cos(angle / two)};
  }
  case (GateName::Rz): {
    auto angle = angles[0];
    return {std::exp(-im<Scalar> * angle / two), 0, 0,
            std::exp(im<Scalar> * angle / two)};
  }
  case (GateName::R1):
    return {{1., 0.}, {0.0, 0.}, {0.0, 0.0}, std::exp(im<Scalar> * angles[0])};
  case (GateName::U1):
    return {{1., 0.}, {0.0, 0.}, {0.0, 0.0}, std::exp(im<Scalar> * angles[0])};
  case (GateName::U2): {
    Scalar oneOverSqrt2 = 1 / std::sqrt(2.);
    auto phi = angles[0];
    auto lambda = angles[1];
    return {{oneOverSqrt2, 0.},
            -oneOverSqrt2 * std::exp(lambda * nvqir::im<Scalar>),
            oneOverSqrt2 * std::exp(nvqir::im<Scalar> * phi),
            oneOverSqrt2 * std::exp(nvqir::im<Scalar> * (phi + lambda))};
  }
  case (GateName::U3): {
    auto theta = angles[0];
    auto phi = angles[1];
    auto lambda = angles[2];
    return {{std::cos(theta / 2), 0.},
            std::exp(nvqir::im<Scalar> * phi) * std::sin(theta / 2),
            -std::exp(nvqir::im<Scalar> * lambda) * std::sin(theta / 2),
            std::exp(nvqir::im<Scalar> * (phi + lambda)) * std::cos(theta / 2)};
  }
  case (GateName::PhasedRx): {
    Scalar two = 2.;
    auto phi = angles[0];
    auto lambda = angles[1];
    return {{std::cos(phi / two), 0.},
            -nvqir::im<Scalar> * std::exp(-nvqir::im<Scalar> * lambda) *
                std::complex<Scalar>{std::sin(phi / two), 0.},
            -nvqir::im<Scalar> * std::exp(nvqir::im<Scalar> * lambda) *
                std::sin(phi / two),
            std::cos(phi / two)};
  }
  }

  throw std::runtime_error("Invalid gate provided to getGateByName.");
}

/// @brief The X operation as a type. Can instantiate and request
/// its matrix data.
template <typename ScalarType = double>
struct x {
  auto getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::X);
  }
  const std::string name() const { return "x"; }
};

/// The Y Gate
template <typename ScalarType = double>
struct y {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::Y);
  }
  const std::string name() const { return "y"; }
};

/// The Z Gate
template <typename ScalarType = double>
struct z {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::Z);
  }
  const std::string name() const { return "z"; }
};

/// The Hadamard Gate
template <typename ScalarType = double>
struct h {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::H);
  }
  const std::string name() const { return "h"; }
};

/// The S Gate
template <typename ScalarType = double>
struct s {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::S);
  }
  const std::string name() const { return "s"; }
};

/// The T Gate
template <typename ScalarType = double>
struct t {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::T);
  }
  const std::string name() const { return "t"; }
};

/// The Sdg Gate
template <typename ScalarType = double>
struct sdg {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::Sdg);
  }
  const std::string name() const { return "sdg"; }
};

/// The Tdg Gate
template <typename ScalarType = double>
struct tdg {
  std::vector<ComplexT<ScalarType>>
  getGate(std::vector<ScalarType> angles = {}) {
    return getGateByName<ScalarType>(GateName::Tdg);
  }
  const std::string name() const { return "tdg"; }
};

/// The RX Rotation Gate
template <typename ScalarType = double>
struct rx {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::Rx, {angles[0]});
  }
  const std::string name() const { return "rx"; }
};

/// The RY Rotation Gate
template <typename ScalarType = double>
struct ry {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::Ry, {angles[0]});
  }
  const std::string name() const { return "ry"; }
};

/// The RZ Rotation Gate
template <typename ScalarType = double>
struct rz {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::Rz, {angles[0]});
  }
  const std::string name() const { return "rz"; }
};

/// @brief The R1 operation as a type. Arbitrary rotation about |1>
template <typename ScalarType = double>
struct r1 {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::R1, {angles[0]});
  }
  const std::string name() const { return "r1"; }
};

/// @brief The U1 operation as a type. Arbitrary rotation about |1>
/// (IBMs version)
template <typename ScalarType = double>
struct u1 {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::U1, {angles[0]});
  }
  const std::string name() const { return "u1"; }
};

template <typename ScalarType = double>
struct u2 {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::U2, {angles[0], angles[1]});
  }
  const std::string name() const { return "u2"; }
};

template <typename ScalarType = double>
struct u3 {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::U3,
                                     {angles[0], angles[1], angles[2]});
  }
  const std::string name() const { return "u3"; }
};

template <typename ScalarType = double>
struct phased_rx {
  std::vector<ComplexT<ScalarType>> getGate(std::vector<ScalarType> angles) {
    return getGateByName<ScalarType>(GateName::PhasedRx,
                                     {angles[0], angles[1]});
  }
  const std::string name() const { return "phased_rx"; }
};

} // namespace nvqir

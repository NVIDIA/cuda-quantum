/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nvqir/Gates.h"

#include <vector>

namespace nvqir {

/// @brief Enumeration of supported CUDA-Q operations
enum class PhotonicGateName {
  PlusGate,
  BeamSplitterGate,
  PhaseShiftGate,
};

/// @brief Returns a precomputed factorial for n up tp 30
double _fast_factorial(int n) {
  static std::vector<double> FACTORIAL_TABLE = {
      1.,
      1.,
      2.,
      6.,
      24.,
      120.,
      720.,
      5040.,
      40320.,
      362880.,
      3628800.,
      39916800.,
      479001600.,
      6227020800.,
      87178291200.,
      1307674368000.,
      20922789888000.,
      355687428096000.,
      6402373705728000.,
      121645100408832000.,
      2432902008176640000.,
      51090942171709440000.,
      1124000727777607680000.,
      25852016738884976640000.,
      620448401733239439360000.,
      15511210043330985984000000.,
      403291461126605635584000000.,
      10888869450418352160768000000.,
      304888344611713860501504000000.,
      8841761993739701954543616000000.,
      265252859812191058636308480000000.,
  };
  if (n > 30) // We do not expect to get 30 photons in the loop at the same time
  {
    throw std::invalid_argument("received invalid value, n <= 30");
  }
  return FACTORIAL_TABLE[n];
}

/// @brief Computes the kronecker delta of two values
int _kron(int a, int b) {
  if (a == b)
    return 1;
  else
    return 0;
}

/// @brief Computes a single element in the matrix representing a beam
/// splitter gate
double _calc_beam_splitter_elem(int N1, int N2, int n1, int n2, double theta) {
  const double t = std::cos(theta); // transmission coefficient
  const double r = std::sin(theta); // reflection coefficient
  double sum = 0;
  for (int k = 0; k <= n1; ++k) {
    int l = N1 - k;
    if (l >= 0 && l <= n2) {
      double term1 = std::pow(r, (n1 - k + l)) * std::pow(t, (n2 + k - l));
      if (term1 == 0) {
        continue;
      }
      double term2 = std::pow((-1), (l)) *
                     (sqrt(_fast_factorial(n1)) * sqrt(_fast_factorial(n2)) *
                      sqrt(_fast_factorial(N1)) * sqrt(_fast_factorial(N2)));
      double term3 = (_fast_factorial(k) * _fast_factorial(n1 - k) *
                      _fast_factorial(l) * _fast_factorial(n2 - l));
      double term = term1 * term2 / term3;
      sum += term;
    } else {
      continue;
    }
  } // end for k

  return sum;
}

/// @brief Computes matrix representing a beam splitter gate
template <typename Scalar>
void _calc_beam_splitter(std::vector<std::complex<Scalar>> &BS,
                         const Scalar theta) {
  int levels = sqrt(sqrt(BS.size()));
  //     """Returns a matrix representing a beam splitter
  for (int n1 = 0; n1 < levels; ++n1) {
    for (int n2 = 0; n2 < levels; ++n2) {
      int nxx = n1 + n2;
      int nxd = std::min(nxx + 1, levels);
      for (int N1 = 0; N1 < nxd; ++N1) {
        int N2 = nxx - N1;
        if (N2 >= nxd) {
          continue;
        } else {

          BS.at(n1 * levels * levels * levels + n2 * levels * levels +
                N1 * levels + N2) =
              _calc_beam_splitter_elem(N1, N2, n1, n2, theta);
        }
      } // end for N1
    } // end for n2
  } // end for n1
}

/// @brief Given the gate name (an element of the GateName enum),
/// return the matrix data, optionally parameterized by a rotation angle.
template <typename Scalar>
std::vector<std::complex<Scalar>>
getPhotonicGateByName(PhotonicGateName name, const std::size_t levels,
                      std::vector<Scalar> angles = {}) {
  switch (name) {
  case (PhotonicGateName::PlusGate): {
    auto length = levels * levels;
    std::vector<std::complex<Scalar>> u(length, 0.0);
    u.at(levels - 1) = 1.;
    for (std::size_t i = 1; i < levels; i++) {
      u.at(i * levels + (i - 1)) = 1.;
    }
    return u;
  }
  case (PhotonicGateName::BeamSplitterGate): {
    auto theta = angles[0];
    auto length = levels * levels * levels * levels;
    std::vector<std::complex<Scalar>> BS(length, 0.0);
    _calc_beam_splitter<Scalar>(BS, theta);
    return BS;
  }
  case (PhotonicGateName::PhaseShiftGate): {

    auto phi = angles[0];
    auto length = levels * levels;
    std::vector<std::complex<Scalar>> PS(length, 0.0);
    // static constexpr std::complex<double> im = std::complex<double>(0, 1.);
    for (std::size_t i = 0; i < levels; i++) {
      PS.at(i * levels + i) =
          std::exp(nvqir::im<Scalar> * static_cast<Scalar>(i) * phi);
    }
    return PS;
  }
  }

  throw std::runtime_error("Invalid gate provided to getGateByName.");
}

/// @brief The plus operation as a type. Can instantiate and request
/// its matrix data.
template <typename ScalarType = double>
struct plus {
  auto getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::PlusGate,
                                             levels);
  }
  const std::string name() const { return "plus"; }
};

/// The Beam Splitter Gate
template <typename ScalarType = double>
struct beam_splitter {
  std::vector<ComplexT<ScalarType>>
  getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::BeamSplitterGate,
                                             levels, angles);
  }
  const std::string name() const { return "beam_splitter"; }
};

/// The Phase Shift Gate
template <typename ScalarType = double>
struct phase_shift {
  std::vector<ComplexT<ScalarType>>
  getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::PhaseShiftGate,
                                             levels, angles);
  }
  const std::string name() const { return "phase_shift"; }
};

} // namespace nvqir

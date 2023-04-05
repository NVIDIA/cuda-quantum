/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "state.h"
#include <iostream>

namespace cudaq {

void state::dump() { dump(std::cout); }
void state::dump(std::ostream &os) {
  auto &[shape, stateData] = data;
  if (shape.size() == 1) {
    for (auto &d : stateData)
      os << d.real() << " ";
    os << "\n";
  } else {
    for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
        os << stateData[i * shape[0] + j].real() << " ";
      }
      os << "\n";
    }
  }
}
std::complex<double> state::operator[](std::size_t idx) {
  auto &[shape, stateData] = data;
  if (shape.size() != 1)
    throw std::runtime_error("Cannot request 1-d index into density matrix. "
                             "Must be a state vector.");
  return stateData[idx];
}

std::complex<double> state::operator()(std::size_t idx, std::size_t jdx) {
  auto &[shape, stateData] = data;

  if (shape.size() != 2)
    throw std::runtime_error("Cannot request 2-d index into state vector. "
                             "Must be a density matrix.");

  return stateData[idx * std::get<0>(data)[0] + jdx];
}

double state::overlap(state &other) {
  double sum = 0.0;
  auto &[shape, stateData] = data;
  if (shape.size() != std::get<0>(other.data).size())
    throw std::runtime_error(
        "Cannot compare state vectors and density matrices.");

  if (shape.size() == 1) {
    for (std::size_t i = 0; i < std::get<1>(data).size(); i++) {
      sum += std::abs(std::get<1>(data)[i] * other[i]);
    }
  } else {

    // Create rho and sigma matrices
    Eigen::MatrixXcd rho =
        Eigen::Map<Eigen::MatrixXcd>(stateData.data(), shape[0], shape[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        std::get<1>(other.data).data(), shape[0], shape[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    sum = (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  // return the overlap
  return sum;
}

// Eigen::MatrixXcd state::as_eigen() {
//   auto &[shape, stateData] = data;
//   if (shape.size() == 1) {
//     // Build up a state vector.
//     // TODO: FIXME
//     return Eigen::Map<Eigen::MatrixXcd>(stateData.data(), shape[0], shape[1])[0];
//   } else {
//     // Build up a density matrix.
//     return Eigen::Map<Eigen::MatrixXcd>(stateData.data(), shape[0], shape[1]);
//   }
// }


} // namespace cudaq

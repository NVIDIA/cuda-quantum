/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Logger.h"
#include "common/SimulationState.h"
#include "qpp.h"

namespace cudaq {

/// @brief QppDmState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Density Matrix Circuit Simulator.
struct QppDmState : public cudaq::SimulationState {
  /// @brief The state.
  qpp::cmat state;

  QppDmState(qpp::cmat &&data) : state(std::move(data)) {}
  QppDmState(const std::vector<std::size_t> &shape,
             const std::vector<std::complex<double>> &data) {
    if (shape.size() != 2)
      throw std::runtime_error(
          "QppDmState must be created from data with 2D shape.");

    state = Eigen::Map<qpp::ket>(const_cast<cudaq::complex *>(data.data()),
                                 shape[0], shape[1]);
  }
  std::size_t getNumQubits() const override { return std::log2(state.rows()); }

  std::vector<std::size_t> getDataShape() const override {
    return {static_cast<std::size_t>(state.rows()),
            static_cast<std::size_t>(state.cols())};
  }

  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        reinterpret_cast<cudaq::complex *>(other.ptr()),
        other.getDataShape()[0], other.getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(const std::vector<cudaq::complex128> &other) override {
    if (other.size() != getDataShape()[0] * getDataShape()[1])
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma =
        Eigen::Map<Eigen::MatrixXcd>(const_cast<cudaq::complex *>(other.data()),
                                     getDataShape()[0], getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(const std::vector<complex64> &data) override {
    throw std::runtime_error(
        "qpp dm state vector requires FP64 data for overlap computation.");
  }

  double overlap(complex128 *other, std::size_t numElements) override {

    if (getNumElements() != numElements)
      throw std::runtime_error("[qpp-dm-state] overlap with pointer data, "
                               "invalid number of elements.");

    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma =
        Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<cudaq::complex *>(other),
                                     getDataShape()[0], getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(cudaq::complex64 *data, std::size_t numElements) override {
    throw std::runtime_error("[qpp-dm-state] overlap pointer requires FP64 "
                             "data for overlap computation.");
  }

  cudaq::complex matrixElement(std::size_t i, std::size_t j) override {
    return state(i, j);
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void *ptr() const override {
    return reinterpret_cast<void *>(const_cast<cudaq::complex *>(state.data()));
  }

  void destroyState() override {
    cudaq::info("qpp-dm-state destroying state vector handle.");
    qpp::cmat k;
    state = k;
  }
};

} // namespace cudaq
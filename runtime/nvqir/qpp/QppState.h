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

/// @brief QppState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Circuit Simulator.
struct QppState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  QppState(qpp::ket &&data) : state(std::move(data)) {}
  QppState(const std::vector<std::size_t> &shape,
           const std::vector<std::complex<double>> &data) {
    if (shape.size() != 1)
      throw std::runtime_error(
          "QppState must be created from data with 1D shape.");

    state =
        Eigen::Map<qpp::ket>(const_cast<complex128 *>(data.data()), shape[0]);
  }

  std::size_t getNumQubits() const override { return std::log2(state.size()); }

  std::vector<std::size_t> getDataShape() const override {
    return {static_cast<std::size_t>(state.size())};
  }

  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    return std::abs(state.transpose()
                        .dot(Eigen::Map<qpp::ket>(
                            reinterpret_cast<complex128 *>(other.ptr()),
                            other.getDataShape()[0]))
                        .real());
  }

  double overlap(const std::vector<cudaq::complex128> &data) override {
    if (data.size() != getDataShape()[0])
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    return std::abs(state.transpose()
                        .dot(Eigen::Map<qpp::ket>(
                            reinterpret_cast<complex128 *>(
                                const_cast<complex128 *>(data.data())),
                            data.size()))
                        .real());
  }

  double overlap(const std::vector<complex64> &data) override {
    throw std::runtime_error(
        "qpp state vector requires FP64 data for overlap computation.");
  }

  double overlap(cudaq::complex128 *data, std::size_t numElements) override {
    if (state.size() != static_cast<Eigen::Index>(numElements))
      throw std::runtime_error(
          "[qpp-state] overlap with pointer data, invalid number of elements.");

    return std::abs(
        state.transpose()
            .dot(Eigen::Map<qpp::ket>(reinterpret_cast<complex128 *>(data),
                                      getDataShape()[0]))
            .real());
  }

  double overlap(cudaq::complex64 *data, std::size_t numElements) override {
    throw std::runtime_error(
        "qpp state vector requires FP64 data for overlap computation.");
  }

  complex128 vectorElement(std::size_t idx) override { return state[idx]; }

  void dump(std::ostream &os) const override { os << state << "\n"; }
  void *ptr() const override {
    return reinterpret_cast<void *>(const_cast<complex128 *>(state.data()));
  }
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    cudaq::info("qpp-state destroying state vector handle.");
    qpp::ket k;
    state = k;
  }
};

} // namespace cudaq
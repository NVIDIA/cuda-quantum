/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CircuitSimulator.h"
#include "Gates.h"
#include "qpp.h"
#include <iostream>

namespace nvqir {

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  /// The QPP state representation (qpp::ket or qpp::cmat)
  StateType state;

  /// Convert from little endian to big endian.
  std::size_t bigEndian(const int n_qubits, const int bit) {
    return n_qubits - bit - 1;
  }

  /// @brief Compute the expectation value <Z...Z> over the given qubit indices.
  /// @param qubit_indices
  /// @return expectation
  double
  calculateExpectationValue(const std::vector<std::size_t> &qubit_indices) {
    const auto hasEvenParity =
        [](std::size_t x,
           const std::vector<std::size_t> &in_qubitIndices) -> bool {
      size_t count = 0;
      for (const auto &bitIdx : in_qubitIndices) {
        if (x & (1ULL << bitIdx)) {
          count++;
        }
      }
      return (count % 2) == 0;
    };

    // First, convert all of the qubit indices to big endian.
    std::vector<std::size_t> casted_qubit_indices;
    for (auto index : qubit_indices) {
      casted_qubit_indices.push_back(bigEndian(nQubitsAllocated, index));
    }

    double result = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
#pragma omp parallel for reduction(+ : result)
      for (std::size_t i = 0; i < stateDimension; ++i) {
        result += (hasEvenParity(i, casted_qubit_indices) ? 1.0 : -1.0) *
                  std::norm(state[i]);
      }
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
#pragma omp parallel for reduction(+ : result)
      for (Eigen::Index i = 0; i < state.rows(); i++) {
        auto element = diag(i).real();
        if (!hasEvenParity(i, casted_qubit_indices))
          element *= -1.;
        result += element;
      }
    }

    return result;
  }

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nTargets) {
    auto nRows = (1UL << nTargets);
    assert(data.size() == nRows * nRows &&
           "Invalid number of gate matrix elements passed to toQppMatrix");

    // we represent row major, they represent column major
    return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>(
        const_cast<std::complex<double> *>(data.data()), nRows, nRows);
  }


  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
    // Update the state vector
    if (state.size() == 0) {
      // If this is the first time, allocate the state
      state = qpp::ket::Zero(stateDimension);
      state(0) = 1.0;
    } else {
      // If we are resizing an existing, allocate
      // a zero state on a single qubit, and Kron-prod
      // that with the existing state.
      qpp::ket zero_state = qpp::ket::Zero(2);
      zero_state(0) = 1.0;
      state = qpp::kron(state, zero_state);
    }
  }

  /// @brief Reset the qubit state.
  void resetQubitStateImpl() override {
    StateType tmp;
    state = tmp;
  }

  void applyGate(const GateApplicationTask &task) override {
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    state = qpp::applyCTRL(state, matrix, task.controls, task.targets);
  }

public:
  QppCircuitSimulator() = default;
  virtual ~QppCircuitSimulator() = default;

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  std::vector<std::size_t> allocateQubits(const std::size_t count) override {
    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++)
      qubits.emplace_back(tracker.getNextIndex());

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      nQubitsAllocated += count;
      stateDimension = calculateStateDim(nQubitsAllocated);
      state = qpp::ket::Zero(stateDimension);
      state(0) = 1.0;
      return qubits;
    }

    nQubitsAllocated += count;
    stateDimension = calculateStateDim(nQubitsAllocated);

    // If we are resizing an existing, allocate
    // a zero state on a n qubit, and Kron-prod
    // that with the existing state.
    qpp::ket zero_state = qpp::ket::Zero((1UL << count));
    zero_state(0) = 1.0;
    state = qpp::kron(state, zero_state);

    return qubits;
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t qubitIdx) override {
    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple =
        qpp::measure(state, qpp::cmat::Identity(2, 2), {qubitIdx},
                     /*qudit dimension=*/2, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.size());
    } else {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.rows(),
                                          collapsed_state.cols());
    }
    cudaq::info("Measured qubit {} -> {}", qubitIdx, measurement_result);
    return measurement_result == 1 ? true : false;
  }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override {
    flushGateQueue();
    state = qpp::reset(state, {qubitIdx});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
    double expectationValue = calculateExpectationValue(measuredBits);
    if (shots < 1) {
      cudaq::info("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstring;
    cudaq::ExecutionResult counts(expectationValue);

    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstring << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      counts.appendResult(bitstring.str(), count);

      // Reset the state.
      bitstring.str("");
      bitstring.clear();
    }

    return counts;
  }

  cudaq::State getStateData() override {
    flushGateQueue();
    // There has to be at least one copy
    return cudaq::State{{stateDimension},
                        {state.data(), state.data() + state.size()}};
  }

  /// @brief Primarily used for testing.
  auto getStateVector() {
    flushGateQueue();
    return state;
  }
  std::string name() const override { return "qpp"; }
  NVQIR_SIMULATOR_CLONE_IMPL(QppCircuitSimulator<StateType>)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::QppCircuitSimulator<qpp::ket>, qpp)
#endif

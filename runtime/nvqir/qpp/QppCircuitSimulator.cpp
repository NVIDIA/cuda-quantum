/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "Gates.h"
#include "qpp.h"
#include <iostream>
#include <set>

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

    std::vector<double> resultVec;
    double result = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      resultVec.resize(stateDimension);
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (std::size_t i = 0; i < stateDimension; ++i) {
        resultVec[i] = (hasEvenParity(i, casted_qubit_indices) ? 1.0 : -1.0) *
                       std::norm(state[i]);
      }
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
      resultVec.resize(state.rows());
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (Eigen::Index i = 0; i < state.rows(); i++) {
        auto element = diag(i).real();
        if (!hasEvenParity(i, casted_qubit_indices))
          element *= -1.;
        resultVec[i] = element;
      }
    }

    // Accumulate outside the for loop to ensure repeatability
    result = std::accumulate(resultVec.begin(), resultVec.end(), 0.0);

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

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t count) override {
    if (count == 0)
      return;

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      state = qpp::ket::Zero(stateDimension);
      state(0) = 1.0;
      return;
    }

    // If we are resizing an existing, allocate
    // a zero state on a n qubit, and Kron-prod
    // that with the existing state.
    qpp::ket zero_state = qpp::ket::Zero((1UL << count));
    zero_state(0) = 1.0;
    state = qpp::kron(state, zero_state);

    return;
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    StateType tmp;
    state = tmp;
  }

  void applyGate(const GateApplicationTask &task) override {
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    state = qpp::applyCTRL(state, matrix, task.controls, task.targets);
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
    state = qpp::ket::Zero(stateDimension);
    state(0) = 1.0;
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

public:
  QppCircuitSimulator() = default;
  virtual ~QppCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    qpp::RandomDevices::get_instance().get_prng().seed(seed);
  }

  void applyExpPauli(double theta, const std::vector<std::size_t> &qubitIds,
                     const cudaq::spin_op &op) override {
    flushGateQueue();
    cudaq::info(" [qpp decomposing] exp_pauli({}, {})", theta,
                op.to_string(false));
    std::vector<std::size_t> qubitSupport;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitSupport.push_back(qubitIds[qubitIdx]);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          rx(!reverse ? M_PI_2 : -M_PI_2, qubitIds[qubitIdx]);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { h(qubitIds[qubitIdx]); });
    });

    if (!basisChange.empty())
      for (auto &basis : basisChange)
        basis(false);

    std::vector<std::pair<std::size_t, std::size_t>> toReverse;
    for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
      x({qubitSupport[i]}, qubitSupport[i + 1]);
      toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
    }

    rz(theta, qubitSupport.back());

    std::reverse(toReverse.begin(), toReverse.end());
    for (auto &[i, j] : toReverse)
      x({i}, j);

    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);
    }
  }

  bool canHandleObserve() override {
    // Do not compute <H> from matrix if shots based sampling requested
    if (executionContext &&
        executionContext->shots != static_cast<std::size_t>(-1)) {
      return false;
    }

    return !shouldObserveFromSampling();
  }

  cudaq::ExecutionResult observe(const cudaq::spin_op &op) override {

    flushGateQueue();

    // The op is on the following target bits.
    std::vector<std::size_t> targets;
    op.for_each_term([&](cudaq::spin_op &term) {
      term.for_each_pauli(
          [&](cudaq::pauli p, std::size_t idx) { targets.push_back(idx); });
    });

    std::sort(targets.begin(), targets.end());
    const auto last_iter = std::unique(targets.begin(), targets.end());
    targets.erase(last_iter, targets.end());

    // Get the matrix as an Eigen matrix
    auto matrix = op.to_matrix();
    qpp::cmat asEigen =
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>>(
            matrix.data(), matrix.rows(), matrix.cols());

    // Compute the expected value
    double ee = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      qpp::ket k = qpp::apply(state, asEigen, targets, 2);
      ee = state.dot(k).real();
    } else {
      ee = qpp::apply(asEigen, state, targets).trace().real();
    }

    return cudaq::ExecutionResult({}, ee);
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
    if (shots < 1) {
      double expectationValue = calculateExpectationValue(measuredBits);
      cudaq::info("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstring;
    cudaq::ExecutionResult counts;

    // Expectation value from the counts
    double expVal = 0.0;
    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstring << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      counts.appendResult(bitstring.str(), count);
      auto par = cudaq::sample_result::has_even_parity(bitstring.str());
      auto p = count / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
      // Reset the state.
      bitstring.str("");
      bitstring.clear();
    }

    counts.expectationValue = expVal;
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

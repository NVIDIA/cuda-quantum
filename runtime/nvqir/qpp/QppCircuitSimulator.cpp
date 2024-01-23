/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"

#include <bit>
#include <iostream>
#include <qpp.h>
#include <set>

namespace nvqir {

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

    state = Eigen::Map<qpp::ket>(const_cast<cudaq::complex *>(data.data()),
                                 shape[0]);
  }

  std::size_t getNumQubits() const override { return std::log2(state.size()); }

  std::vector<std::size_t> getDataShape() const override {
    return {static_cast<std::size_t>(state.size())};
  }

  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    return state.transpose()
        .dot(Eigen::Map<qpp::ket>(
            reinterpret_cast<cudaq::complex *>(other.ptr()),
            other.getDataShape()[0]))
        .real();
  }

  double overlap(const std::vector<cudaq::complex> &data) override {
    if (data.size() != getDataShape()[0])
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    return state.transpose()
        .dot(
            Eigen::Map<qpp::ket>(reinterpret_cast<cudaq::complex *>(
                                     const_cast<cudaq::complex *>(data.data())),
                                 data.size()))
        .real();
  }

  double overlap(void *data) override {
    return state.transpose()
        .dot(Eigen::Map<qpp::ket>(reinterpret_cast<cudaq::complex *>(data),
                                  getDataShape()[0]))
        .real();
  }

  cudaq::complex vectorElement(std::size_t idx) override { return state[idx]; }

  void dump(std::ostream &os) const override { os << state << "\n"; }
  void *ptr() const override {
    return reinterpret_cast<void *>(const_cast<cudaq::complex *>(state.data()));
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

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  /// The QPP state representation (qpp::ket or qpp::cmat)
  StateType state;

  /// @brief Convert internal qubit index to Q++ qubit index.
  ///
  /// In Q++, qubits are indexed from left to right, and thus q0 is the leftmost
  /// qubit. Internally, in CUDA Quantum, qubits are index from right to left,
  /// hence q0 is the rightmost qubit. Example:
  /// ```
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 3  2  1  0 : CUDA Quantum indices
  /// ```
  std::size_t convertQubitIndex(std::size_t qubitIndex) {
    assert(stateDimension > 0 && "The state is empty, and thus has no qubits");
    return std::log2(stateDimension) - qubitIndex - 1;
  }

  /// @brief Compute the expectation value <Z...Z> over the given qubit indices.
  double calculateExpectationValue(const std::vector<std::size_t> &qubits) {
    std::size_t bitmask = 0;
    for (auto q : qubits)
      bitmask |= (1ULL << q);
    const auto hasEvenParity = [&bitmask](std::size_t x) -> bool {
      return std::popcount(x & bitmask) % 2 == 0;
    };

    std::vector<double> result;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      result.resize(stateDimension);
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (std::size_t i = 0; i < stateDimension; ++i)
        result[i] = (hasEvenParity(i) ? 1.0 : -1.0) * std::norm(state[i]);
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
      result.resize(state.rows());
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (Eigen::Index i = 0; i < state.rows(); ++i)
        result[i] = hasEvenParity(i) ? diag(i).real() : -diag(i).real();
    }

    // Accumulate outside the for loop to ensure repeatability
    return std::accumulate(result.begin(), result.end(), 0.0);
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
  void addQubitToState() override { addQubitsToState(1); }

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
    state = qpp::kron(zero_state, state);

    return;
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    StateType tmp;
    state = tmp;
  }

  void applyGate(const GateApplicationTask &task) override {
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    // First, convert all of the qubit indices to big endian.
    std::vector<std::size_t> controls;
    for (auto index : task.controls) {
      controls.push_back(convertQubitIndex(index));
    }
    std::vector<std::size_t> targets;
    for (auto index : task.targets) {
      targets.push_back(convertQubitIndex(index));
    }

    if (controls.empty()) {
      state = qpp::apply(state, matrix, targets);
      return;
    }
    state = qpp::applyCTRL(state, matrix, controls, targets);
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
    state = qpp::ket::Zero(stateDimension);
    state(0) = 1.0;
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    const auto qubitIdx = convertQubitIndex(index);
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

  std::unique_ptr<cudaq::SimulationState> createSimulationState(
      const std::vector<std::size_t> &shape,
      const std::vector<std::complex<double>> &data) const override {
    return std::make_unique<QppState>(shape, data);
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
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    const auto qubitIdx = convertQubitIndex(index);
    state = qpp::reset(state, {qubitIdx});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    if (shots < 1) {
      double expectationValue = calculateExpectationValue(qubits);
      cudaq::info("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    std::vector<std::size_t> measuredBits;
    for (auto index : qubits) {
      measuredBits.push_back(convertQubitIndex(index));
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstream;
    cudaq::ExecutionResult counts;

    // Expectation value from the counts
    double expVal = 0.0;
    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstream << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      auto bitstring = bitstream.str();
      counts.appendResult(bitstring, count);
      auto par = cudaq::sample_result::has_even_parity(bitstring);
      auto p = count / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
      // Reset the state.
      bitstream.str("");
      bitstream.clear();
    }

    counts.expectationValue = expVal;
    return counts;
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();

    if constexpr (std::is_same_v<qpp::ket, StateType>) {
      return std::make_unique<QppState>(std::move(state));
    } else {
      return nullptr;
    }
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
